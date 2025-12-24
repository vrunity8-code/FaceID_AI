"""
3D Face Visualization Utilities
Provides functions to draw 3D mesh, pose axes, and landmarks
"""

import cv2
import numpy as np


def draw_3d_mesh(img, mesh_data, color=(0, 255, 0), thickness=1):
    """
    Draw 3D face mesh wireframe on image
    
    Args:
        img: Input image
        mesh_data: Dictionary with 'vertices_2d' and 'triangles'
        color: Color for mesh lines (B, G, R)
        thickness: Line thickness
        
    Returns:
        Image with mesh drawn
    """
    if mesh_data is None:
        return img
    
    img_out = img.copy()
    vertices_2d = mesh_data['vertices_2d']
    triangles = mesh_data.get('triangles', [])
    
    # Draw triangles
    for tri in triangles:
        if len(tri) == 3:
            pts = vertices_2d[tri].astype(np.int32)
            # Draw triangle edges
            for i in range(3):
                pt1 = tuple(pts[i])
                pt2 = tuple(pts[(i+1) % 3])
                cv2.line(img_out, pt1, pt2, color, thickness)
    
    return img_out


def draw_3d_landmarks(img, landmarks_2d, color=(0, 255, 255), radius=2):
    """
    Draw 3D facial landmarks on image
    
    Args:
        img: Input image
        landmarks_2d: Nx2 array of 2D landmark coordinates
        color: Color for landmarks (B, G, R)
        radius: Radius of landmark points
        
    Returns:
        Image with landmarks drawn
    """
    if landmarks_2d is None:
        return img
    
    img_out = img.copy()
    
    for i, (x, y) in enumerate(landmarks_2d):
        x, y = int(x), int(y)
        
        # Different colors for different facial regions
        if i < 17:  # Face contour
            pt_color = (255, 200, 0)  # Cyan
        elif i < 27:  # Eyebrows
            pt_color = (0, 255, 0)  # Green
        elif i < 36:  # Nose
            pt_color = (0, 200, 255)  # Orange
        elif i < 48:  # Eyes
            pt_color = (255, 0, 0)  # Blue
        else:  # Mouth
            pt_color = (0, 0, 255)  # Red
        
        cv2.circle(img_out, (x, y), radius, pt_color, -1)
        
        # Add landmark index for debugging (optional)
        # cv2.putText(img_out, str(i), (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, pt_color, 1)
    
    return img_out


def draw_pose_axes(img, pose_info, length=50):
    """
    Draw 3D pose axes (X, Y, Z) on face
    
    Args:
        img: Input image
        pose_info: Dictionary with rotation and translation info
        length: Length of axes in pixels
        
    Returns:
        Image with pose axes drawn
    """
    if pose_info is None:
        return img
    
    img_out = img.copy()
    
    # Get rotation and translation
    rotation_vector = pose_info['rotation_vector']
    translation = pose_info['translation'].reshape(3, 1)
    camera_matrix = pose_info['camera_matrix']
    
    # Define 3D axes points
    axis_3d = np.float32([
        [0, 0, 0],      # Origin
        [length, 0, 0], # X-axis (red)
        [0, length, 0], # Y-axis (green)
        [0, 0, length]  # Z-axis (blue)
    ])
    
    # Project 3D axes to 2D
    axis_2d, _ = cv2.projectPoints(
        axis_3d,
        rotation_vector,
        translation,
        camera_matrix,
        np.zeros((4, 1))
    )
    axis_2d = axis_2d.reshape(-1, 2).astype(int)
    
    # Draw axes
    origin = tuple(axis_2d[0])
    
    # X-axis (red)
    cv2.arrowedLine(img_out, origin, tuple(axis_2d[1]), (0, 0, 255), 2, tipLength=0.3)
    cv2.putText(img_out, 'X', tuple(axis_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Y-axis (green)
    cv2.arrowedLine(img_out, origin, tuple(axis_2d[2]), (0, 255, 0), 2, tipLength=0.3)
    cv2.putText(img_out, 'Y', tuple(axis_2d[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Z-axis (blue)
    cv2.arrowedLine(img_out, origin, tuple(axis_2d[3]), (255, 0, 0), 2, tipLength=0.3)
    cv2.putText(img_out, 'Z', tuple(axis_2d[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return img_out


def draw_pose_info(img, pose_info, position=(10, 30)):
    """
    Draw pose information (pitch, yaw, roll) on image
    
    Args:
        img: Input image
        pose_info: Dictionary with pitch, yaw, roll angles
        position: Top-left position for text
        
    Returns:
        Image with pose info drawn
    """
    if pose_info is None:
        return img
    
    img_out = img.copy()
    
    pitch = pose_info.get('pitch', 0)
    yaw = pose_info.get('yaw', 0)
    roll = pose_info.get('roll', 0)
    
    x, y = position
    line_height = 25
    
    # Draw background for better visibility
    overlay = img_out.copy()
    cv2.rectangle(overlay, (x-5, y-20), (x+200, y+line_height*3), (0, 0, 0), -1)
    img_out = cv2.addWeighted(img_out, 0.6, overlay, 0.4, 0)
    
    # Draw pose angles
    cv2.putText(img_out, f'Pitch: {pitch:6.2f}°', (x, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(img_out, f'Yaw:   {yaw:6.2f}°', (x, y + line_height), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(img_out, f'Roll:  {roll:6.2f}°', (x, y + line_height*2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return img_out


def draw_face_mesh_overlay(img, face, mesh_data, pose_info, 
                           show_mesh=True, show_landmarks=True, 
                           show_axes=True, show_pose_text=True):
    """
    Draw complete 3D face visualization overlay
    
    Args:
        img: Input image
        face: Face object from detector
        mesh_data: 3D mesh data
        pose_info: Pose estimation data
        show_mesh: Whether to show mesh wireframe
        show_landmarks: Whether to show 3D landmarks
        show_axes: Whether to show pose axes
        show_pose_text: Whether to show pose angle text
        
    Returns:
        Image with complete visualization
    """
    img_out = img.copy()
    
    # Draw mesh wireframe
    if show_mesh and mesh_data is not None:
        img_out = draw_3d_mesh(img_out, mesh_data, color=(0, 200, 0), thickness=1)
    
    # Draw 3D landmarks
    if show_landmarks and mesh_data is not None:
        landmarks_2d = mesh_data.get('landmarks_2d')
        img_out = draw_3d_landmarks(img_out, landmarks_2d, radius=2)
    
    # Draw pose axes
    if show_axes and pose_info is not None:
        img_out = draw_pose_axes(img_out, pose_info, length=100)
    
    # Draw pose angle text
    if show_pose_text and pose_info is not None:
        bbox = face.bbox.astype(int)
        text_pos = (bbox[0], max(10, bbox[1] - 10))
        img_out = draw_pose_info(img_out, pose_info, position=text_pos)
    
    return img_out


def visualize_face_contour(img, landmarks_2d):
    """
    Draw face contour and key facial features
    
    Args:
        img: Input image
        landmarks_2d: 68 facial landmarks in 2D
        
    Returns:
        Image with facial features highlighted
    """
    if landmarks_2d is None:
        return img
    
    img_out = img.copy()
    landmarks = landmarks_2d.astype(np.int32)
    
    # Face contour (0-16)
    cv2.polylines(img_out, [landmarks[0:17]], False, (255, 200, 0), 2)
    
    # Right eyebrow (17-21)
    cv2.polylines(img_out, [landmarks[17:22]], False, (0, 255, 0), 2)
    
    # Left eyebrow (22-26)
    cv2.polylines(img_out, [landmarks[22:27]], False, (0, 255, 0), 2)
    
    # Nose bridge (27-30)
    cv2.polylines(img_out, [landmarks[27:31]], False, (0, 200, 255), 2)
    
    # Nose bottom (31-35)
    cv2.polylines(img_out, [landmarks[31:36]], False, (0, 200, 255), 2)
    
    # Right eye (36-41)
    cv2.polylines(img_out, [landmarks[36:42]], True, (255, 0, 0), 2)
    
    # Left eye (42-47)
    cv2.polylines(img_out, [landmarks[42:48]], True, (255, 0, 0), 2)
    
    # Outer mouth (48-59)
    cv2.polylines(img_out, [landmarks[48:60]], True, (0, 0, 255), 2)
    
    # Inner mouth (60-67)
    cv2.polylines(img_out, [landmarks[60:68]], True, (0, 0, 200), 2)
    
    return img_out
