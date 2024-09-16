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


def create_pointcloud(path_to_txt_file):
    """
    Erzeugt eine Punktwolke (PointCloud) aus einer .txt-Datei und weist ihr Positionen, Farben und semantische Labels zu.

    Args:
        path_to_txt_file (str): Pfad zur .txt-Datei, die die Punktinformationen enthält. 
            Die Datei sollte die folgenden Spalten enthalten:
            - Spalte 0-2: x, y, z-Koordinaten (Positionen der Punkte).
            - Spalte 3-5: r, g, b-Werte (Farbe der Punkte).
            - Spalte 7: Semantisches Label der Punkte.

    Returns:
        open3d.t.geometry.PointCloud: Eine PointCloud mit gefüllten Punkten, Farben und semantischen Labels.
    """
    txt_file_positions = np.loadtxt(path_to_txt_file, usecols = (0, 1, 2), dtype=np.float64)
    txt_file_attributes = np.loadtxt(path_to_txt_file, usecols = (3, 4, 5, 7), dtype=np.uint8)
    pcd = o3d.t.geometry.PointCloud() # erstellt eine leere PointCloud
    pcd.point.positions = txt_file_positions[:, 0:3] # fuellt sie mit Punkten
    pcd.point.colors = txt_file_attributes[:, 0:3] # fuellt sie mit Attribut: Farbe
    pcd.point.sem_label = txt_file_attributes[:, 3] # fuellt sie mit Attribut: semantischen Label
    print(f"pcd.point_sem_label: {pcd.point.sem_label}")
    return pcd

def visualize_pointcloud(pcd):
    o3d.visualization.draw_geometries([pcd.to_legacy()],
                                    zoom=0.3412,
                                    front=[0.4257, -0.2125, -0.8795],
                                    lookat=[2.6172, 2.0475, 1.532],
                                    up=[-0.0694, -0.9768, 0.2024])


path_to_txt_files = "/home/bluebell/repositories/bachelorarbeit/forks/pointnet/sem_seg/log6/dump"
txt_file_name = "Area_6_lounge_1_pred.txt"
path_to_txt_file = os.path.join(path_to_txt_files, txt_file_name)


pcd = create_pointcloud(path_to_txt_file)
#visualize_pointcloud(pcd)
#print(pcd)

# dbscan
inst_label = 1 # zaehlvariable fuer die instanzlabel
inst_label_minus_one = 0

num_of_clusters = 13

pcd_asnumpy = np.loadtxt(path_to_txt_file, usecols = (0, 1, 2, 3, 4, 5, 7))

inst_seg_pcd_asnumpy = np.empty((0, pcd_asnumpy.shape[1]+1))
print(f"inst_seg_pcd_asnumpy.shape: {inst_seg_pcd_asnumpy.shape}")

for label in range(num_of_clusters): # wir iterieren ueber jedes label
    print(f"Label: {label}")
    print()
    last_column = pcd_asnumpy[:, 6]
    mask = last_column.astype(int) == label

    part_pcd_asnumpy = pcd_asnumpy[mask, :] # only pointcloud data with the same semantic label
    print(f"part_pcd_asnumpy.shape: {part_pcd_asnumpy.shape}")

    # if label == 0: 
    #     print(mask)
    #     print(mask.sum())
    #     print(part_pcd_asnumpy.shape)

    part_pcd =  o3d.t.geometry.PointCloud() # erstellt eine leere PointCloud
    part_pcd.point.positions = part_pcd_asnumpy[:, 0:3]
    labels = part_pcd.cluster_dbscan(eps=0.4, min_points=10, print_progress=True)
    labels = labels.numpy()
    # if 12 in labels:
    #     print("True")
    print(f"labels of dbscan: {labels}")
    if len(labels) > 0: 
        labels[labels.astype(int) == -1] = inst_label_minus_one
        max_label = labels.max()
        print(f"max_label: {labels.max()}")
        #print(f"labels: {type(labels)}")
        print(f"labels.shape: {labels.shape}")
        for lbl in range(max_label+1):
            labels[labels.astype(int) == lbl] = inst_label
            inst_label += 1
        result = np.concatenate((part_pcd_asnumpy, np.reshape(labels, (len(labels), 1))), axis=1)
        inst_seg_pcd_asnumpy = np.vstack((inst_seg_pcd_asnumpy, result))
        #labels[labels.astype(int) == -1] = -1
        
# print(f"inst_seg_pcd_asnumpy: {inst_seg_pcd_asnumpy}")
# print(f"pcd_asnumpy.shape: {pcd_asnumpy.shape}")
# print(f"inst_seg_pcd_asnumpy.shape: {inst_seg_pcd_asnumpy.shape}")
# print(f"inst_label: {inst_label}")

# write results into a txt file

file_path = os.path.join(path_to_txt_files, txt_file_name[:-4])
file_path += "_extended.txt"
# with open(file_path, "w") as file:
#     file.write

np.savetxt(file_path, inst_seg_pcd_asnumpy)

# create pointcloud
pcd_extended = o3d.t.geometry.PointCloud() # erstellt eine leere PointCloud
pcd_extended.point.positions = inst_seg_pcd_asnumpy[:, 0:3] # fuellt sie mit Punkten
pcd_extended.point.colors = (inst_seg_pcd_asnumpy[:, 3:6]).astype(np.uint8) # fuellt sie mit Attribut: Farbe
pcd_extended.point.sem_label = (inst_seg_pcd_asnumpy[:, 6]).astype(np.uint8) # fuellt sie mit Attribut: semantischen Label
pcd_extended.point.inst_label = (inst_seg_pcd_asnumpy[:, 7]).astype(np.uint8)
    #print(f"labels:{labels}")
#visualize_pointcloud(pcd_extended)

# visualization of instance segmentation
file_path = os.path.join(path_to_txt_files, txt_file_name[:-4])
file_path += "_extended.ply"
# o3d.io.write_point_cloud(filename=file_path, pointcloud=pcd_extended.to_legacy(), write_ascii=True)

max_label = labels.max().item()
# print(f"point cloud has {max_label + 1} clusters")
# print(f"inst_label: {inst_label}")
cmap = plt.get_cmap("viridis", inst_label+1)
colors = cmap(np.arange(inst_label))
colors = (colors[:, :3] * 255).astype(np.uint8)
#print(f"colors: {colors}")
print(f"colors.shape: {colors.shape}")
# colors = o3c.Tensor(colors[:, :3], o3c.float32)
# colors[labels < 0] = 0
# pcd_extended.point.colors = colors

pcd_extended = o3d.t.geometry.PointCloud() # erstellt eine leere PointCloud
pcd_extended.point.positions = inst_seg_pcd_asnumpy[:, 0:3] # fuellt sie mit Punkten


inst_labels = (inst_seg_pcd_asnumpy[:, 7]).astype(np.uint8)

# print(f"inst_labels: {inst_labels}")
# print("Anfang")
# for lb in inst_labels:
#     if lb > 80:
#         print("inst_label ist groesser 80")
#         print(f"lb: {lb}")
# print(f"Ende")

pcd_extended.point.colors = colors[inst_labels, :]

#print(f"pcd_extended.point.colors: {pcd_extended.point.colors}")
#(inst_seg_pcd_asnumpy[:, 3:6]).astype(np.uint8) # fuellt sie mit Attribut: Farbe
# pcd_extended .point.sem_label = (inst_seg_pcd_asnumpy[:, 6]).astype(np.uint8) # fuellt sie mit Attribut: semantischen Label
# pcd_extended .point.inst_label = (inst_seg_pcd_asnumpy[:, 7]).astype(np.uint8)

o3d.visualization.draw_geometries([pcd_extended.to_legacy()],
                                  zoom=0.455,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])

        
    








