import os
import numpy as np
import open3d as o3d

def visualize_pointcloud(pcd, zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172, 2.0475, 1.532], up=[-0.0694, -0.9768, 0.2024]):
    """
    Visualisiert eine Punktwolke (PointCloud) mit Open3D.

    Args:
        pcd (open3d.t.geometry.PointCloud): Die zu visualisierende Punktwolke.
        zoom (float, optional): Der Zoom-Faktor für die Ansicht. Standard ist 0.3412.
        front (list, optional): Die Front-Richtung für die Kameraansicht. Standard ist [0.4257, -0.2125, -0.8795].
        lookat (list, optional): Der Punkt, auf den die Kamera schaut. Standard ist [2.6172, 2.0475, 1.532].
        up (list, optional): Die Aufwärtsrichtung der Kamera. Standard ist [-0.0694, -0.9768, 0.2024].

    Returns:
        None
    """    
    o3d.visualization.draw_geometries([pcd],
                                    zoom=zoom,
                                    front=front,
                                    lookat=lookat,
                                    up=up)

path_s3dis_gt = "./data/Stanford3dDataset_v1.2_Aligned_Version"
sample_path = "Area_6/lounge_1"
sample_annotations = "Annotations"

path_gt_inst_seg = os.path.join(path_s3dis_gt, sample_path, sample_annotations)

# Liste ueber alle txt files erstellen
txt_files = []
for element in os.listdir(path_gt_inst_seg):
    if element.endswith(".txt"):
        txt_files.append(element)

cls_names = [
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter"
]
# numpy array fuer Ergebnis erstellen
pcd_np = np.empty((0, 8))
inst_label_counter = 0 

full_num_pts = 0

sem_labels_list = []

for inst_file in txt_files: # inst_file enthaelt xyzrgb + Klasse im Namen
    semantic_label, _ = inst_file.split("_")#inst_file[:-4]
    semantic_label_id = cls_names.index(semantic_label)
    instance = []
    # print(f"inst_file: {inst_file}")
    # print(f"semantic_label: {semantic_label}")
    # print(f"semantic_label_id: {semantic_label_id}")
    path_file = os.path.join(path_gt_inst_seg, inst_file)
    inst_data = np.loadtxt(path_file)
    num_pts = inst_data.shape[0]
    full_num_pts += num_pts
    #print(f"num_pts: {num_pts}")
    #print(f"inst_data: {inst_data}")
    column_sem_label = np.full((num_pts, 1), semantic_label_id)
    column_inst_label = np.full((num_pts, 1), inst_label_counter)
    inst_data = np.hstack((inst_data, column_sem_label, column_inst_label))
    #print(f"inst_data: {inst_data}")
    pcd_np = np.vstack((pcd_np, inst_data))
    sem_labels_list.append(semantic_label)
    inst_label_counter += 1

save_path = "./data/Area_1_lounge_1_gt.txt"
#np.savetxt(save_path, pcd_np)
print(f"sem_label_list: {sem_labels_list}")

# visualization
x_min, x_max = np.min(pcd_np[:, 0]), np.max(pcd_np[:, 0])
y_min, y_max = np.min(pcd_np[:, 1]), np.max(pcd_np[:, 1])
z_min, z_max = np.min(pcd_np[:, 2]), np.max(pcd_np[:, 2])

center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]

color_instance = np.array([31, 119, 180]).astype(np.uint8) # [ 31 119 180] blauton
color_rest = np.array([171, 173, 161]).astype(np.uint8) # grau-gelber Ton    #[158 218 229]
colors = np.array([color_rest, color_instance])
instances = set(pcd_np[:, 7].astype(np.uint8).flatten())
# Schritt 1: gehe jedes Objekt durch
for instance_label in instances:
    print(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(f"instance_label: {instance_label}")
    #print(f"instance_label: {instance_label}")
#   Schritt 2: es sollte bereits zwei feste Farben geben für Instanz mit Label x und für den Rest der Punkte y
#   Schritt 3: erzeuge eine Open3D Datenstruktur 
    pcd = o3d.t.geometry.PointCloud() # erstellt eine leere PointCloud
    pcd.point.positions = pcd_np[:, 0:3] # fuellt sie mit Punkten
    color_picker = (pcd_np[:, 7] == instance_label) # .astype(np.uint8)
    point_colors = np.tile(colors[0], (pcd_np.shape[0], 1))
    point_colors[color_picker] = colors[1]
    pcd.point.colors = point_colors
#   Schritt 4: gebe allen Punkten die Farbe Y
#   Schritt 5: gebe allen Punkten mit Instanzlabel x die Farbe X
#   Schritt 6: Visualisiere die Pointcloud
    sem_label = ((pcd_np[color_picker, 6])[0]).astype(np.uint8)
    print(f"sem_label: {sem_label}")
    print(f"cls_name: {cls_names[sem_label]}")
    visualize_pointcloud(pcd.to_legacy(), zoom=0.5, front=[0, -1, -1], lookat=center, up=[0, 1, 0])



