import pyrealsense2 as rs
import cv2
import numpy as np

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

def print_intrinsics():
    color_profile = profile.get_stream(rs.stream.color)
    depth_profile = profile.get_stream(rs.stream.depth)
    
    color_intrin = color_profile.as_video_stream_profile().get_intrinsics()
    depth_intrin = depth_profile.as_video_stream_profile().get_intrinsics()
    
    print("\n=== Color Camera Intrinsics ===")
    print(f"Resolution: {color_intrin.width}x{color_intrin.height}")
    print(f"Focal Length: fx={color_intrin.fx:.2f}, fy={color_intrin.fy:.2f}")
    print(f"Principal Point: ({color_intrin.ppx:.2f}, {color_intrin.ppy:.2f})")
    print(f"Distortion Model: {color_intrin.model}")
    print(f"Distortion Coefficients (k1, k2, p1, p2, k3): {color_intrin.coeffs}")

    print("\n=== Depth Camera Intrinsics ===")
    print(f"Resolution: {depth_intrin.width}x{depth_intrin.height}")
    print(f"Focal Length: fx={depth_intrin.fx:.2f}, fy={depth_intrin.fy:.2f}")
    print(f"Principal Point: ({depth_intrin.ppx:.2f}, {depth_intrin.ppy:.2f})")
    print(f"Distortion Model: {depth_intrin.model}")
    print(f"Distortion Coefficients (k1, k2, p1, p2, k3): {depth_intrin.coeffs}")

print_intrinsics()

align_to = rs.stream.color
align = rs.align(align_to)

is_recording = False
video_writer = None

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )

        cv2.imshow('RGB', color_image)
        cv2.imshow('Depth', depth_colormap)

        key = cv2.waitKey(1)

        if key == ord(' '):
            timestamp = cv2.getTickCount()
            cv2.imwrite(f'rgb_{timestamp}.png', color_image)
            print(f"Saved snapshot {timestamp}")

        if key == ord('r'):
            if not is_recording:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 480))
                is_recording = True
                print("Recording started")
            else:
                video_writer.release()
                is_recording = False
                print("Recording stopped")

        if is_recording:
            video_writer.write(color_image)

        if key == ord('q'):
            break

finally:
    pipeline.stop()
    if is_recording:
        video_writer.release()
    cv2.destroyAllWindows()
    print("Pipeline stopped")
