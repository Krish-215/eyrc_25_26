#!/usr/bin/env python3
"""
Step 4 (from your reference): 2D detection + depth read at green-top centroid.

- Uses your detection code and thresholds.
- Subscribes to RGB and Depth topics (tries common topic names).
- For each accepted fruit, reads median depth in a small patch around the green-top centroid,
  converts units correctly (uint16 -> mm -> m), and displays depth text next to the centroid.
- Logs depths for debugging.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

# ------------------ User thresholds (kept from your provided code) ------------------
LOWER_GRAY = np.array([0, 0, 50])
UPPER_GRAY = np.array([180, 50, 220])

LOWER_GREEN = np.array([35, 60, 60])
UPPER_GREEN = np.array([85, 255, 255])

# Detection params
MIN_CONTOUR_AREA = 200
MAX_CONTOUR_AREA = 5000
TOP_REGION_HEIGHT_RATIO = 0.30
MIN_GREEN_PIXELS_ABS = 700
MIN_GREEN_RATIO = 0.01    # fraction of top region area

# Depth reading params
DEPTH_MEDIAN_KERNEL = 9    # patch size (7x7)
MIN_DEPTH = 0.01           # meters
MAX_DEPTH = 3.0            # meters

# Topics (tries multiple common names)
RGB_TOPICS = ['/camera/image_raw']
DEPTH_TOPICS = ['/camera/depth/image_raw']

WINDOW_NAME = "Step4: Depth at Green Centroid"

class Step4DepthFromRef(Node):
    def __init__(self):
        super().__init__('step4_depth_from_ref')
        self.bridge = CvBridge()
        self.cv_image = None
        self.depth_image = None

        # subscribe to first available RGB topic
        self.rgb_topic_used = None
        for t in RGB_TOPICS:
            try:
                self.create_subscription(Image, t, self.rgb_cb, 10)
                self.rgb_topic_used = t
                self.get_logger().info(f"Subscribed to RGB topic: {t}")
                break
            except Exception:
                continue
        if self.rgb_topic_used is None:
            self.get_logger().warning("No RGB topic found among candidates. Please check topic names.")

        # subscribe to first available depth topic
        self.depth_topic_used = None
        for t in DEPTH_TOPICS:
            try:
                self.create_subscription(Image, t, self.depth_cb, 10)
                self.depth_topic_used = t
                self.get_logger().info(f"Subscribed to Depth topic: {t}")
                break
            except Exception:
                continue
        if self.depth_topic_used is None:
            self.get_logger().warning("No Depth topic found among candidates. Please check topic names.")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        self.create_timer(0.05, self.tick)  # 20 Hz timer for visualization

    def rgb_cb(self, msg: Image):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warning(f"RGB conversion failed: {e}")
            self.cv_image = None

    def depth_cb(self, msg: Image):
        try:
            # keep original encoding; could be uint16 (mm) or float32 (m)
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warning(f"Depth conversion failed: {e}")
            self.depth_image = None

    # median depth read around (x,y)
    def read_depth_median(self, x, y, k=DEPTH_MEDIAN_KERNEL):
        if self.depth_image is None:
            return None
        try:
            h, w = self.depth_image.shape[:2]
            px = int(np.clip(x, 0, w - 1))
            py = int(np.clip(y, 0, h - 1))
            r = k // 2
            x0 = max(0, px - r); x1 = min(w - 1, px + r)
            y0 = max(0, py - r); y1 = min(h - 1, py + r)
            patch = self.depth_image[y0:y1+1, x0:x1+1]
            if patch.size == 0:
                return None

            # handle different dtypes
            if patch.dtype == np.uint16:
                patch_f = patch.astype(np.float32)
                patch_f[patch_f == 0] = np.nan
                depth_m = float(np.nanmedian(patch_f)) / 1000.0  # mm -> m
            elif patch.dtype in [np.float32, np.float64]:
                patch_f = patch.astype(np.float32)
                patch_f[patch_f <= 0.0] = np.nan
                depth_m = float(np.nanmedian(patch_f))
            else:
                patch_f = patch.astype(np.float32)
                patch_f[patch_f == 0] = np.nan
                depth_m = float(np.nanmedian(patch_f))

            if math.isnan(depth_m) or depth_m < MIN_DEPTH or depth_m > MAX_DEPTH:
                return None
            return depth_m
        except Exception as e:
            self.get_logger().warning(f"Depth read error at ({x},{y}): {e}")
            return None

    def tick(self):
        if self.cv_image is None:
            # show empty image window so user sees something
            cv2.imshow(WINDOW_NAME, np.zeros((10,10,3), dtype=np.uint8))
            cv2.waitKey(1)
            return

        img = self.cv_image.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # masks
        mask_gray = cv2.inRange(hsv, LOWER_GRAY, UPPER_GRAY)
        mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

        # morphological clean on grey
        kernel = np.ones((3,3), np.uint8)
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_OPEN, kernel)
        mask_gray = cv2.morphologyEx(mask_gray, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fruit_id = 1
        H, W = mask_gray.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA):
                continue

            x,y,w,h = cv2.boundingRect(cnt)

            # draw grey contour + bbox
            cv2.drawContours(img, [cnt], -1, (0,0,255), 2)
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

            # body centroid
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                body_cx = int(M["m10"]/M["m00"])
                body_cy = int(M["m01"]/M["m00"])
            else:
                body_cx, body_cy = x+w//2, y+h//2
            cv2.circle(img, (body_cx, body_cy), 4, (255,255,255), -1)

            # TOP REGION
            y_top_start = y
            y_top_end = y + int(h * TOP_REGION_HEIGHT_RATIO)
            y_top_end = min(y_top_end, H)
            top_region = mask_green[y_top_start:y_top_end, x:x+w]

            green_pixel_count = np.sum(top_region > 0)
            top_area = max(1, top_region.size)
            min_green_needed = max(MIN_GREEN_PIXELS_ABS, int(MIN_GREEN_RATIO * top_area))

            # draw top region box
            cv2.rectangle(img, (x, y_top_start), (x+w, y_top_end), (255,255,0), 2)

            if green_pixel_count < min_green_needed:
                cv2.putText(img, f"{fruit_id}: NO GREEN", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                fruit_id += 1
                continue

            # green centroid in top region
            Mg = cv2.moments(top_region)
            if Mg["m00"] != 0:
                gx = int(Mg["m10"]/Mg["m00"])
                gy = int(Mg["m01"]/Mg["m00"])
            else:
                ys, xs = np.where(top_region > 0)
                if len(xs) == 0:
                    fruit_id += 1
                    continue
                gx = int(np.mean(xs))
                gy = int(np.mean(ys))

            # convert to full-image coordinates
            gx_full = x + gx
            gy_full = y_top_start + gy

            # draw green-top centroid marker
            cv2.circle(img, (gx_full, gy_full), 6, (255,0,0), -1)
            cv2.putText(img, f"{fruit_id}: GREEN OK", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # read depth at green centroid
            depth_m = self.read_depth_median(gx_full, gy_full, k=DEPTH_MEDIAN_KERNEL)

            if depth_m is None:
                cv2.putText(img, "NO DEPTH", (gx_full + 8, gy_full - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
                self.get_logger().info(f"ID:{fruit_id} - NO VALID DEPTH at px ({gx_full},{gy_full})")
            else:
                cv2.putText(img, f"{depth_m:.2f} m", (gx_full + 8, gy_full - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                self.get_logger().info(f"ID:{fruit_id} - depth={depth_m:.3f} m at px ({gx_full},{gy_full}) green_px={green_pixel_count}")

            # draw sampled patch rectangle (for debug)
            k = DEPTH_MEDIAN_KERNEL // 2
            x0 = max(0, gx_full - k); x1 = min(W-1, gx_full + k)
            y0 = max(0, gy_full - k); y1 = min(H-1, gy_full + k)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0,255,255), 1)

            fruit_id += 1

        # display
        try:
            cv2.imshow(WINDOW_NAME, img)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warning(f"OpenCV display error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = Step4DepthFromRef()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
