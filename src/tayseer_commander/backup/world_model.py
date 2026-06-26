# world_model_node.py
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from tayseer_interfaces.srv import GetObjectLocation, ListObjects, UpdateObject
import json
import os
import sqlite3
from datetime import datetime
from threading import Lock
import numpy as np


class WorldModelNode(Node):
    def __init__(self):
        super().__init__('world_model')

        self.declare_parameter("match_distance", 0.3)
        self.declare_parameter("alpha", 0.3)
        self.match_distance = self.get_parameter("match_distance").value
        self.alpha = self.get_parameter("alpha").value

        self.world_state_pub = self.create_publisher(String, '/world_state', 10)
        self.create_timer(1.0, self.pub_world_state)

        self.objs = {}
        self.lock = Lock()
        self.class_counters = {} 

        self.db_path = os.path.expanduser('~/tayseer_ws/world_model.db')
        self._init_db()
        self.load_from_db()

        self.create_subscription(PoseStamped,'/detected_object',self.obj_det_cb,10)

        self.get_loc_srv = self.create_service(GetObjectLocation,'/get_object_location',self.handle_get_loc)
        self.list_srv = self.create_service(ListObjects,'/list_objects',self.handle_list_objs)
        self.update_srv = self.create_service(UpdateObject,'/update_object',self.handle_upd_obj)

        self.get_logger().info(f"World Model ready (match_distance={self.match_distance}m, alpha={self.alpha})")

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("PRAGMA table_info(objects)")
            columns = [row[1] for row in cursor.fetchall()]
            if columns and ('instance_id' not in columns or 'class_name' not in columns):
                self.get_logger().warn("Old DB schema found; dropping table.")
                conn.execute("DROP TABLE objects")
                conn.commit()

            conn.execute('''
                CREATE TABLE IF NOT EXISTS objects (
                    instance_id TEXT PRIMARY KEY,
                    class_name TEXT,
                    pos_x REAL,
                    pos_y REAL,
                    pos_z REAL,
                    frame_id TEXT,
                    last_seen TEXT,
                    confidence REAL
                )
            ''')
            conn.commit()

    def _upsert_obj(self, instance_id, obj):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO objects
                (instance_id, class_name, pos_x, pos_y, pos_z,
                 frame_id, last_seen, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                instance_id,
                obj['class_name'],
                obj['position'][0],
                obj['position'][1],
                obj['position'][2],
                obj['frame_id'],
                obj['last_seen'],
                obj['confidence'],
            ))
            conn.commit()

    def _gen_inst_id(self, class_name: str) -> str:
        if class_name not in self.class_counters:
            max_idx = -1
            for inst_id in self.objs.keys():
                if inst_id.startswith(f"{class_name}_"):
                    try:
                        idx = int(inst_id.split('_')[-1])
                        max_idx = max(max_idx, idx)
                    except ValueError:
                        pass
            self.class_counters[class_name] = max_idx + 1
        idx = self.class_counters[class_name]
        self.class_counters[class_name] += 1
        return f"{class_name}_{idx}"

    def obj_det_cb(self, msg: PoseStamped):
        class_name = msg.header.frame_id
        new_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])

        with self.lock:
            best_id = None
            best_dist = float('inf')

            for inst_id, obj in self.objs.items():
                if obj['class_name'] != class_name:
                    continue
                dist = float(np.linalg.norm(np.array(obj['position']) - new_pos))
                if dist < best_dist:
                    best_dist = dist
                    best_id = inst_id

            if best_id is not None and best_dist < self.match_distance:
                old_pos = np.array(self.objs[best_id]['position'])
                smoothed_pos = self.alpha * new_pos + (1.0 - self.alpha) * old_pos

                self.objs[best_id]['position'] = [
                    float(smoothed_pos[0]),
                    float(smoothed_pos[1]),
                    float(smoothed_pos[2]),
                ]
                self.objs[best_id]['last_seen'] = datetime.now().isoformat()
                self._upsert_obj(best_id, self.objs[best_id])
                self.get_logger().info(
                    f"Updated {best_id} at ({smoothed_pos[0]:.2f}, {smoothed_pos[1]:.2f}, {smoothed_pos[2]:.2f})"
                )
            else:
                instance_id = self._gen_inst_id(class_name)
                obj_data = {
                    'class_name': class_name,
                    'position': [
                        float(new_pos[0]),
                        float(new_pos[1]),
                        float(new_pos[2]),
                    ],
                    'frame_id': 'map',
                    'last_seen': datetime.now().isoformat(),
                    'confidence': 0.95,
                }
                self.objs[instance_id] = obj_data
                self._upsert_obj(instance_id, obj_data)
                self.get_logger().info(
                    f"Added {instance_id} at ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f})"
                )

    def handle_get_loc(self, req, res):
        with self.lock:
            obj = self.objs.get(req.obj_name)
            if obj:
                res.found = True
                res.position.x = obj['position'][0]
                res.position.y = obj['position'][1]
                res.position.z = obj['position'][2]
                res.frame_id = obj['frame_id']
            else:
                res.found = False
        return res

    def handle_list_objs(self, req, res):
        with self.lock:
            res.obj_names = list(self.objs.keys())
            res.count = len(res.obj_names)
        return res

    def handle_upd_obj(self, req, res):
        with self.lock:
            if req.obj_name in self.objs:
                class_name = self.objs[req.obj_name]['class_name']
            else:
                parts = req.obj_name.rsplit('_', 1)
                class_name = parts[0] if len(parts) == 2 and parts[1].isdigit() else req.obj_name

            obj_data = {
                'class_name': class_name,
                'position': [
                    req.position.x,
                    req.position.y,
                    req.position.z,
                ],
                'frame_id': req.frame_id,
                'last_seen': datetime.now().isoformat(),
                'confidence': req.confidence,
            }
            self.objs[req.obj_name] = obj_data
            self._upsert_obj(req.obj_name, obj_data)
            res.success = True
        return res

    def load_from_db(self):
        if not os.path.exists(self.db_path):
            return
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        'SELECT instance_id, class_name, pos_x, pos_y, pos_z, '
                        'frame_id, last_seen, confidence FROM objects'
                    )
                    for row in cursor.fetchall():
                        inst_id, cls, x, y, z, frame_id, last_seen, conf = row
                        self.objs[inst_id] = {
                            'class_name': cls,
                            'position': [x, y, z],
                            'frame_id': frame_id,
                            'last_seen': last_seen,
                            'confidence': conf,
                        }
                        if inst_id.startswith(cls + '_'):
                            try:
                                idx = int(inst_id.split('_')[-1])
                                self.class_counters[cls] = max(
                                    self.class_counters.get(cls, 0), idx + 1
                                )
                            except ValueError:
                                pass
            self.get_logger().info(f"Loaded {len(self.objs)} objects from DB")
        except Exception as e:
            self.get_logger().warn(f"Failed to load world model from DB: {e}")

    def pub_world_state(self):
        with self.lock:
            msg = String()
            msg.data = json.dumps(self.objs)
            self.world_state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WorldModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()