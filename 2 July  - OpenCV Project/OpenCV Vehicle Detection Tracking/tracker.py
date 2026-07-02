import math


class EuclideanDistTracker:
    def __init__(self):
        # Store center positions of tracked objects
        self.center_points = {}

        # ID counter
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []

        # Process detected objects
        for rect in objects_rect:
            x, y, w, h = rect

            # Calculate center point
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Check if the object has already been detected
            same_object_detected = False

            for object_id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 25:
                    self.center_points[object_id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, object_id])
                    same_object_detected = True
                    break

            # Assign a new ID if it's a new object
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Remove IDs that are no longer detected
        new_center_points = {}

        for _, _, _, _, object_id in objects_bbs_ids:
            new_center_points[object_id] = self.center_points[object_id]

        self.center_points = new_center_points.copy()

        return objects_bbs_ids