import os
import numpy as np
import open3d as o3d
import inst_seg_utilities as isu

# Schritt 0: Laden der semantisch segmentierten Punktwolkendaten (xyz + rgb + semantischer Label)
# Schritt 1: Aufruf isu.inst_seg_with_dbscan zur Anwendung der Instanzsegmentierung auf die Punktwolke
# Schritt 2: Aufruf isu.postprocess_instance_segmentation 
# zur Visualisierung Methoden wie isu.visualize_pointcloud oder isu.vis_instance_seg_results aufrufen
# Ergebnisse der pointcloud werden als txt file abgespeichert: mit Endung _extended.txt (Ergebnis ohne Postprocessing),
# mit Endung _extended_postprocessed.txt mit Postprocessing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
txt_dir_path = os.path.join(BASE_DIR, "log6/dump")
txt_file_name = "Area_6_lounge_1_pred.txt"
txt_file_path = os.path.join(txt_dir_path, txt_file_name)


pcd = isu.create_pointcloud_from_txt_file(txt_file_path) 
isu.visualize_pointcloud(pcd.to_legacy()) # Visualisierung der Pointcloud mit korrekten Farben   
# load data from txt file in a numpy array and run instance segmentation (inst_seg-with-dbscan)
pcd_np = np.loadtxt(txt_file_path, usecols = (0, 1, 2, 3, 4, 5, 7))
inst_seg_pcd_np, inst_label_counter = isu.inst_seg_with_dbscan(pcd_np, min_points=10, eps=0.4)

# store results as txt file
file_path = os.path.join(txt_dir_path, txt_file_name[:-4])
file_path_inst_seg_result = file_path + "_extended.txt"
np.savetxt(file_path_inst_seg_result, inst_seg_pcd_np) # wieder einkommentieren

isu.vis_instance_seg_results(inst_label_counter, inst_seg_pcd_np) # Visualisierung Pointcloud mit Instanzsegmentierung (ohne Postprocessing)


#print(f"inst_label_counter: {inst_label_counter}")
# oid = inst_seg_pcd_np[:, 7].astype(np.uint8)
# number_oid = len(set(oid.flatten()))
#print(f"oid: {len(oid)}")
#print(f"number of oids: {number_oid}")

# Schritt 0: entferne noise und outliers, renumbere die Instanzlabels von 0 bis (#Instanzen - 1)
inst_seg_pcd_np_postprocessed = isu.postprocess_instance_segmentation(inst_seg_pcd_np, noise_label=0, outlier_removal_threshold=200)

# inst_ids = inst_seg_pcd_np_postprocessed[:, 7].astype(np.uint8)
# number_unique_instances = len(set(inst_ids.flatten()))
# print(f"inst_ids: {set(inst_ids.flatten())}")
# print(f"number_unique_instances: {number_unique_instances}") 

# Visualisierung der Postprocessed PointCloud

# Visualisierung basierend auf den einzelnen semantischen Klassen
## Beginn Visualisierung einzelner semantischer Klassen
sem_labels = inst_seg_pcd_np_postprocessed[:, 6].astype(np.uint8)
sem_labels = sorted(set(sem_labels.flatten()))

for sem_label in sem_labels: 
    part_pcd_np = inst_seg_pcd_np_postprocessed[inst_seg_pcd_np_postprocessed[:, 6]==sem_label, :]
    renumbered_part_pcd_np = isu.renumber_instance_ids(part_pcd_np, start_inst_label=0)
    inst_ids = renumbered_part_pcd_np[:, 7].astype(np.uint8)
    unique_inst_ids = set(inst_ids.flatten())
    number_unique_instances = len(unique_inst_ids)
    #print(f"unique_inst_ids: {unique_inst_ids}")
    #print(f"number_unique_instances: {number_unique_instances}")
    isu.vis_instance_seg_results(inst_label_counter=number_unique_instances, inst_seg_pcd_np=renumbered_part_pcd_np)
## Ende Visualisierung semantischer Klassen

isu.vis_instance_seg_results(inst_label_counter=len(set(inst_seg_pcd_np_postprocessed[:, 7].astype(np.uint8).flatten())), inst_seg_pcd_np=inst_seg_pcd_np_postprocessed) # Visualisierung postprocessed pointcloud (mit Instanzsegmentierung)
# save pointcloud (instance segmentation + postprocessing)
file_path = os.path.join(txt_dir_path, txt_file_name[:-4])
file_path_inst_seg_result = file_path + "_extended_postprocessed.txt"
#np.savetxt(file_path_inst_seg_result, inst_seg_pcd_np_postprocessed) # einkommentieren, falls man postprocessed pointcloud speichern möchte

# Visualisierung jedes einzelnen Objekts
# das Objekt bekomme eine Farbe X, alle anderen Punkte bekommen Farbe Y
# Schritt 0: Farbe auswahlen, Instanzen (Labels) ermitteln
cls_names = ["ceiling", "floor", "wall", "beam", "column", "window", "door", "table", "chair", "sofa", "bookcase", "board","clutter"]
color_instance = np.array([31, 119, 180]).astype(np.uint8) # [ 31 119 180] blauton
color_rest = np.array([171, 173, 161]).astype(np.uint8) # grau-gelber Ton    #[158 218 229]
colors = np.array([color_rest, color_instance])
instances = set(inst_seg_pcd_np_postprocessed[:, 7].astype(np.uint8).flatten())
# Schritt 1: gehe jedes Objekt durch
for instance_label in instances:
    print(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(f"instance_label: {instance_label}")
    #print(f"instance_label: {instance_label}")
#   Schritt 2: es sollte bereits zwei feste Farben geben für Instanz mit Label x und für den Rest der Punkte y
#   Schritt 3: erzeuge eine Open3D Datenstruktur 
    pcd = o3d.t.geometry.PointCloud() # erstellt eine leere PointCloud
    pcd.point.positions = inst_seg_pcd_np_postprocessed[:, 0:3] # fuellt sie mit Punkten
    color_picker = (inst_seg_pcd_np_postprocessed[:, 7] == instance_label) # .astype(np.uint8)
    point_colors = np.tile(colors[0], (inst_seg_pcd_np_postprocessed.shape[0], 1))
    point_colors[color_picker] = colors[1]
    pcd.point.colors = point_colors
#   Schritt 4: gebe allen Punkten die Farbe Y
#   Schritt 5: gebe allen Punkten mit Instanzlabel x die Farbe X
#   Schritt 6: Visualisiere die Pointcloud
    sem_label = ((inst_seg_pcd_np_postprocessed[color_picker, 6])[0]).astype(np.uint8)
    print(f"sem_label: {sem_label}")
    print(f"cls_name: {cls_names[sem_label]}")
    isu.visualize_pointcloud(pcd.to_legacy())
#   Schritt 7: Zeige auch das semantische Label (als kleine Orientierung)
# Ende Visualisierung jedes einzelnen Objekts