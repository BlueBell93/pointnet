# Ziel ist die Instanzsegmentierung einer Punktwolke
# hierfuer wird mittels PointNet zuerst semantisch segmentiert (dies ist bereits geschehen)
# anschließend wird DBSCAN angewendet
# DBSCAN wird zur Instanztrennung innerhalb einer Kategorie verwendet, da DBSCAN nur auf die Punkte derselben semantischen
# Kategorie angewendet werden soll
# zum Schluss sollen die Punkte mit ihrem Instanzlabel wieder zusammengefuehrt werden
# das ganze geschieht auf dem bereits semantische segmentierten s3dis Datensatz (area 6, Testdatensatz)
# txt Datei enthätl x y z (normalisiert) r g b pred_val pred_label (von 0 bis 12)

# Schritt 1: einlesen der txt Datei (gemacht)
# Schritt 2: konvertieren in ply file (xyz rgb pred_valu pred_label) (gemacht)
# Schritt 3: gehe ueber jede der n semantischen Kategorien
#   Schritt 3.1: wende dbscan an
#   Schritt 3.2: verwalte die Ergebnisse
# Schritt 4: bilde wieder eine einzige Punktwolke (hier muss darauf geachtet werden, die labels konsistent zu halten)


import os
import numpy as np
import open3d as o3d
import open3d.core as o3c
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_pointcloud_from_txt_file(txt_file_path):
    """
    Erzeugt eine Punktwolke (PointCloud) aus einer .txt-Datei und weist ihr Positionen, Farben und semantische Labels zu.

    Args:
        txt_file_path (str): Pfad zur .txt-Datei, die die Punktinformationen enthält. 
            Die Datei sollte die folgenden Spalten enthalten:
            - Spalte 0-2: x, y, z-Koordinaten (Positionen der Punkte).
            - Spalte 3-5: r, g, b-Werte (Farbe der Punkte).
            - Spalte 7: Semantisches Label der Punkte.

    Returns:
        open3d.t.geometry.PointCloud: Eine PointCloud mit gefüllten Punkten, Farben und semantischen Labels.
    """
    txt_file_positions = np.loadtxt(txt_file_path, usecols = (0, 1, 2), dtype=np.float64)
    txt_file_attributes = np.loadtxt(txt_file_path, usecols = (3, 4, 5, 7), dtype=np.uint8)
    pcd = o3d.t.geometry.PointCloud() # erstellt eine leere PointCloud
    pcd.point.positions = txt_file_positions[:, 0:3] # fuellt sie mit Punkten
    pcd.point.colors = txt_file_attributes[:, 0:3] # fuellt sie mit Attribut: Farbe
    pcd.point.sem_label = txt_file_attributes[:, 3] # fuellt sie mit Attribut: semantischen Label
    print(f"pcd.point_sem_label: {pcd.point.sem_label}")
    return pcd

def visualize_pointcloud(pcd, zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172, 2.0475, 1.532], up=[-0.0694, -0.9768, 0.2024]):
    o3d.visualization.draw_geometries([pcd],
                                    zoom=zoom,
                                    front=front,
                                    lookat=lookat,
                                    up=up)

def save_pointcloud_to_ply(inst_seg_pcd_np, file_path):
    # create pointcloud
    pcd_extended = o3d.t.geometry.PointCloud() # erstellt eine leere PointCloud
    pcd_extended.point.positions = inst_seg_pcd_np[:, 0:3] # fuellt sie mit Punkten
    pcd_extended.point.colors = (inst_seg_pcd_np[:, 3:6]).astype(np.uint8) # fuellt sie mit Attribut: Farbe
    pcd_extended.point.sem_label = (inst_seg_pcd_np[:, 6]).astype(np.uint8) # fuellt sie mit Attribut: semantischen Label
    pcd_extended.point.inst_label = (inst_seg_pcd_np[:, 7]).astype(np.uint8)
    #visualize_pointcloud(pcd_extended)
    # problem: beim Speichern mit write_point_cloud werden nur positions und colors gespeichert...

    # Visualization of instance segmentation
    file_path = os.path.join(txt_dir_path, txt_file_name[:-4])
    file_path += "_extended.ply"
    # o3d.io.write_point_cloud(filename=file_path, pointcloud=pcd_extended.to_legacy(), write_ascii=True)
    return pcd_extended

# create colors for visualization of the instance segmentation results
def create_colors_for_instance_seg_vis(num_labels):
    #max_label = labels.max().item()
    cmap = plt.get_cmap("viridis", num_labels)
    colors = cmap(np.arange(num_labels))
    colors = (colors[:, :3] * 255).astype(np.uint8)
    #print(f"colors.shape: {colors.shape}")
    return colors

# create pointcloud to show instance segmentation results
def create_pointcloud_for_instance_seg_vis(inst_seg_pcd_np, colors):
    pcd_extended = o3d.t.geometry.PointCloud() # erstellt eine leere PointCloud
    pcd_extended.point.positions = inst_seg_pcd_np[:, 0:3] # fuellt sie mit Punkten
    inst_labels = (inst_seg_pcd_np[:, 7]).astype(np.uint8)
    pcd_extended.point.colors = colors[inst_labels, :]
    return pcd_extended

# visualize instance segmentation results as a pointcloud
def vis_instance_seg_results(inst_label_counter, inst_seg_pcd_np):
    colors = create_colors_for_instance_seg_vis(inst_label_counter+1)
    pcd = create_pointcloud_for_instance_seg_vis(inst_seg_pcd_np, colors)
    visualize_pointcloud(pcd.to_legacy())


def inst_seg_with_dbscan(pcd_np, num_sem_cls=13, eps=0.4, min_points=10):
    inst_label_counter = 1 # zaehlvariable fuer die instanzlabel
    inst_label_noise = 0
    inst_seg_pcd_np = np.empty((0, pcd_np.shape[1]+1)) # stores results

    for semantic_cls in range(num_sem_cls): # wir iterieren ueber jedes semantic_cls
        semantic_labels_column = pcd_np[:, 6]
        mask = semantic_labels_column.astype(int) == semantic_cls

        semantically_filtered_pcd = pcd_np[mask, :] # only pointcloud data with the same semantic semantic_cls

        part_pcd =  o3d.t.geometry.PointCloud() # erstellt eine leere PointCloud
        part_pcd.point.positions = semantically_filtered_pcd[:, 0:3] # fuellt die pointcloud mit xyz-Daten
        labels = part_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True) # dbscan
        labels = labels.numpy()
        #print(f"labels of dbscan: {labels}")
        if len(labels) > 0: 
            labels[labels.astype(int) == -1] = inst_label_noise
            max_label = labels.max()
            #print(f"max_label: {labels.max()}")
            #print(f"labels.shape: {labels.shape}")
            for dbscan_cluster in range(max_label+1):
                labels[labels.astype(int) == dbscan_cluster] = inst_label_counter
                inst_label_counter += 1
            result = np.concatenate((semantically_filtered_pcd, np.reshape(labels, (len(labels), 1))), axis=1)
            inst_seg_pcd_np = np.vstack((inst_seg_pcd_np, result))
            #labels[labels.astype(int) == -1] = -1
    return (inst_seg_pcd_np, labels, inst_label_counter)

txt_dir_path = os.path.join(BASE_DIR, "log6/dump")
txt_file_name = "Area_6_lounge_1_pred.txt"
txt_file_path = os.path.join(txt_dir_path, txt_file_name)

#pcd = create_pointcloud_from_txt_file(txt_file_path)
#visualize_pointcloud(pcd)
        
# load data from txt file in a numpy array and run instance segmentation
pcd_np = np.loadtxt(txt_file_path, usecols = (0, 1, 2, 3, 4, 5, 7))
inst_seg_pcd_np, labels, inst_label_counter = inst_seg_with_dbscan(pcd_np)

# store results as txt file
file_path = os.path.join(txt_dir_path, txt_file_name[:-4])
file_path += "_extended.txt"
np.savetxt(file_path, inst_seg_pcd_np)

vis_instance_seg_results(inst_label_counter, inst_seg_pcd_np)