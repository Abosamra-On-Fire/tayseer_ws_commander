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


class WorldModelNode(Node):
    def __init__(self):
        super().__init__('world_model')
        self.world_state_pub = self.create_publisher(String, '/world_state', 10)
        self.create_timer(1.0, self.publish_world_state)

        # In-memory database: {name: {position: [x,y,z], frame_id: "map", ...}}
        self.objects = {}
        self.lock = Lock()

        # SQLite database path
        self.db_path = os.path.expanduser('~/tayseer_ws/world_model.db')
        self._init_db()
        self.load_from_db()

        # Subscribers
        self.create_subscription(
            PoseStamped,
            '/detected_object',
            self.object_detected_callback,
            10
        )

        # Services (synchronous query/response)
        self.get_loc_srv = self.create_service(
            GetObjectLocation,
            '/get_object_location',
            self.handle_get_location
        )
        self.list_srv = self.create_service(
            ListObjects,
            '/list_objects',
            self.handle_list_objects
        )
        self.update_srv = self.create_service(
            UpdateObject,
            '/update_object',
            self.handle_update_object
        )

        self.get_logger().info("World Model node initialized (SQLite backend)")

    def _init_db(self):
        """Create the SQLite table if it doesn't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS objects (
                    name TEXT PRIMARY KEY,
                    pos_x REAL,
                    pos_y REAL,
                    pos_z REAL,
                    frame_id TEXT,
                    last_seen TEXT,
                    confidence REAL
                )
            ''')
            conn.commit()

    def _upsert_object(self, name, obj):
        """Write or overwrite a single object record in SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO objects 
                (name, pos_x, pos_y, pos_z, frame_id, last_seen, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                name,
                obj['position'][0],
                obj['position'][1],
                obj['position'][2],
                obj['frame_id'],
                obj['last_seen'],
                obj['confidence']
            ))
            conn.commit()

    def object_detected_callback(self, msg: PoseStamped):
        """Called by Perception node when object detected."""
        # Parse object name from header.frame_id or a dedicated field
        # Adjust based on your Perception node's output format
        object_name = msg.header.frame_id  # e.g., "blue_cube"

        with self.lock:
            obj_data = {
                'position': [
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z
                ],
                'frame_id': msg.header.frame_id,
                'last_seen': datetime.now().isoformat(),
                'confidence': 0.95  # Get from perception if available
            }
            self.objects[object_name] = obj_data
            self._upsert_object(object_name, obj_data)

        self.get_logger().info(
            f"Updated {object_name} at "
            f"({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
        )

    def handle_get_location(self, request, response):
        with self.lock:
            obj = self.objects.get(request.object_name)
            if obj:
                response.found = True
                response.position.x = obj['position'][0]
                response.position.y = obj['position'][1]
                response.position.z = obj['position'][2]
                response.frame_id = obj['frame_id']
            else:
                response.found = False
        return response

    def handle_list_objects(self, request, response):
        with self.lock:
            response.object_names = list(self.objects.keys())
            response.count = len(response.object_names)
        return response

    def handle_update_object(self, request, response):
        with self.lock:
            obj_data = {
                'position': [
                    request.position.x,
                    request.position.y,
                    request.position.z
                ],
                'frame_id': request.frame_id,
                'last_seen': datetime.now().isoformat(),
                'confidence': request.confidence
            }
            self.objects[request.object_name] = obj_data
            self._upsert_object(request.object_name, obj_data)
            response.success = True
        return response

    def load_from_db(self):
        """Hydrate the in-memory cache from SQLite at startup."""
        if not os.path.exists(self.db_path):
            return

        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        'SELECT name, pos_x, pos_y, pos_z, frame_id, last_seen, confidence '
                        'FROM objects'
                    )
                    for row in cursor.fetchall():
                        name, x, y, z, frame_id, last_seen, confidence = row
                        self.objects[name] = {
                            'position': [x, y, z],
                            'frame_id': frame_id,
                            'last_seen': last_seen,
                            'confidence': confidence
                        }
            self.get_logger().info(
                f"Loaded {len(self.objects)} objects from SQLite database"
            )
        except Exception as e:
            self.get_logger().warn(f"Failed to load world model from DB: {e}")

    def publish_world_state(self):
        with self.lock:
            msg = String()
            msg.data = json.dumps(self.objects)
            self.world_state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = WorldModelNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()