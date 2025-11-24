import cv2
import torch
import trimesh
import numpy as np
# import plotly.graph_objects as go
from joints_definitions import MHR_JOINT_MAP, MEASUREMENT_CONFIG

def get_tpose_mesh_final(estimator, outputs):
    mhr_layer = estimator.model.head_pose.mhr
    device = next(mhr_layer.parameters()).device
    raw_shape = outputs[0]['shape_params']
    id_coeffs = torch.tensor(raw_shape).float().to(device).unsqueeze(0)
    pose_coeffs = torch.zeros((1, 204)).float().to(device)
    face_coeffs = torch.zeros((1, 72)).float().to(device)
    with torch.no_grad():
        res = mhr_layer(
            id_coeffs,      
            pose_coeffs,    
            face_coeffs,    
            True            
        )
    if isinstance(res, tuple):
        if isinstance(res[0], (tuple, list)):
            verts = res[0][0]
            joints = res[0][1]
        else:
            verts = res[0]
            joints = res[1]
    else:
        verts = res
        joints = None
    try:
        if hasattr(mhr_layer, 'character_torch'):
            faces = mhr_layer.character_torch.mesh.faces
        else:
    
            faces = estimator.model.head_pose.mhr.character_torch.mesh.faces
        faces_np = faces.cpu().numpy()
    except:
        faces_np = None
    verts_np = verts.cpu().numpy()[0]
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np)
    if joints is not None:
        joints_np = joints.cpu().numpy()[0]
    else:
        joints_np = None
    return mesh, joints_np


def scale_model(mesh, joints, target_height_cm):
    current_height = mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min()
    scale_factor = target_height_cm / current_height
    mesh.vertices *= scale_factor
    joints *= scale_factor
    return mesh, joints, scale_factor

def get_largest_slice_loop(mesh, height_y):
    slice_obj = mesh.section(plane_origin=[0, height_y, 0], plane_normal=[0, 1, 0])
    if not slice_obj: return 0.0, None

    max_len = 0.0
    best_path = None
    for loop_points in slice_obj.discrete:
        diffs = np.diff(loop_points, axis=0)
        loop_len = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
        loop_len += np.linalg.norm(loop_points[-1] - loop_points[0])
        
        if loop_len > max_len:
            max_len = loop_len
            best_path = loop_points
    return max_len, best_path

def calculate_body_metrics(mesh, joints, idx_map, config):
    results = {}

    for name, rule in config.items():
        color = rule.get("color", "white")

        # --- A. CIRCUMFERENCE ---
        if rule["type"] == "circumference":
            y_coords = [joints[idx_map[j]][1] for j in rule["landmarks"]]
            target_y = rule["func"](y_coords) + rule["offset"]
            val, path = get_largest_slice_loop(mesh, target_y)
            
            results[name] = {
                'type': 'circ', 'value': val, 'path': path, 'color': color
            }

        # --- B. POLYLINE (Chain of Segments) ---
        elif rule["type"] == "polyline":
            chain_indices = [idx_map[j] for j in rule["landmarks"]]
            chain_points = joints[chain_indices]
            
            # Calculate total length by summing segments
            total_dist = 0.0
            for i in range(len(chain_points) - 1):
                p1 = chain_points[i]
                p2 = chain_points[i+1]
                total_dist += np.linalg.norm(p1 - p2)

            results[name] = {
                'type': 'polyline', # We treat linear and polyline similarly in viz now
                'value': total_dist,
                'points': chain_points,
                'color': color
            }
            
        # --- C. LINEAR (Simple Start-End) ---
        elif rule["type"] == "linear":
            p_start = joints[idx_map[rule["start_joint"]]]
            p_end   = joints[idx_map[rule["end_joint"]]]
            dist = np.linalg.norm(p_start - p_end)
            
            results[name] = {
                'type': 'polyline', # Reuse the polyline logic for visualization
                'value': dist,
                'points': np.array([p_start, p_end]),
                'color': color
            }

    return results

# def create_visualization(mesh, joints, measurements, idx_map, title="Body Analysis"):
#     traces = []

#     # Mesh
#     x, y, z = mesh.vertices.T
#     i, j, k = mesh.faces.T
#     traces.append(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color='lightgray', opacity=0.3, name='Skin'))

#     for name, data in measurements.items():
#         color = data['color']
#         val = data['value']

#         if data['type'] == 'circ' and data['path'] is not None:
#             traces.append(go.Scatter3d(
#                 x=data['path'][:, 0], y=data['path'][:, 1], z=data['path'][:, 2],
#                 mode='lines', line=dict(color=color, width=5), name=f"{name}"
#             ))
        
#         # Unified visualization for Polyline and Linear
#         elif data['type'] == 'polyline':
#             pts = data['points']
#             # Draw the connected line
#             traces.append(go.Scatter3d(
#                 x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
#                 mode='lines+markers', line=dict(color=color, width=5), 
#                 marker=dict(size=4), name=f"{name}"
#             ))
            
#             # Floating Text Label (at the middle point of the chain)
#             mid_idx = len(pts) // 2
#             mid_pt = pts[mid_idx]
            
#             # If it's a 2-point line, find exact geometric center
#             if len(pts) == 2:
#                 mid_pt = np.mean(pts, axis=0)

#             traces.append(go.Scatter3d(
#                 x=[mid_pt[0] + 5], y=[mid_pt[1]], z=[mid_pt[2] + 2],
#                 mode='text', text=[f"{val:.1f}cm"],
#                 textfont=dict(color=color, size=12, family="Arial Black"),
#                 showlegend=False
#             ))

#     # Joints
#     rel_indices = [idx_map[k] for k in idx_map]
#     rel_joints = joints[rel_indices]
#     traces.append(go.Scatter3d(
#         x=rel_joints[:, 0], y=rel_joints[:, 1], z=rel_joints[:, 2],
#         mode='markers', marker=dict(size=4, color='red'), name='Joints'
#     ))

#     fig = go.Figure(data=traces)
#     fig.update_layout(
#         title=title, 
#         scene=dict(aspectmode='data', xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
#         margin=dict(t=40, b=0, l=0, r=0)
#     )
#     return fig

def get_measurements(mesh_orig, joints_orig, target_height_cm, visualize=False):
    results = {}
    print(f"--- ANALYZING (Target: {target_height_cm}cm) ---")
    mesh = mesh_orig.copy()
    joints = joints_orig.copy()
    mesh, joints, _ = scale_model(mesh, joints, target_height_cm)
    measures = calculate_body_metrics(mesh, joints, MHR_JOINT_MAP, MEASUREMENT_CONFIG)
    
    print("\n--- RESULTS ---")
    for k, v in measures.items():
        print(f"{k:<15}: {v['value']:.2f} cm")
        results[k] = v['value']

    if visualize:
        fig = create_visualization(mesh, joints, measures, MHR_JOINT_MAP, title=f"Curved Analysis")
        fig.show()
    
    return results