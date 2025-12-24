import numpy as np

class SimpleTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 0
        self.objects = {} # id -> centroid
        self.disappeared = {} # id -> frames_disappeared
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """
        Update tracker with new bounding boxes.
        rects: list of [x1, y1, x2, y2]
        Returns: dict of object_id -> rect
        """
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}

        # Calculate centroids
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Calculate distances between input centroids and existing object centroids
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        # Return dictionary of ID -> BBox
        # We need to map back which rect belongs to which ID
        # This simple tracker tracks centroids. 
        # To return bboxes, we need to match the updated centroids back to input rects.
        # For simplicity in this specific use case, we will return a list of (ID, rect) tuples
        # by re-associating the closest input rect to the tracked object.
        
        tracked_objects = []
        
        # Re-match for output
        # This is a bit redundant but ensures we return the fresh bbox from the detector
        for object_id, centroid in self.objects.items():
            # Find closest input rect
            best_rect = None
            min_dist = float('inf')
            
            for rect in rects:
                cX = int((rect[0] + rect[2]) / 2.0)
                cY = int((rect[1] + rect[3]) / 2.0)
                dist = np.linalg.norm(centroid - np.array([cX, cY]))
                if dist < min_dist:
                    min_dist = dist
                    best_rect = rect
            
            if best_rect is not None and min_dist < self.max_distance:
                tracked_objects.append((object_id, best_rect))
                
        return tracked_objects
