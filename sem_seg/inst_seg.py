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
# neue ToDos: Noise entfernen, d.h. Instanzen mit wenigen Punkten, da diese nicht aussagekräftig sind, d.h. visuell nicht als Objekt erkennbar sind


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
    #print(f"pcd.point_sem_label: {pcd.point.sem_label}")
    return pcd

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

def save_pointcloud_to_ply(inst_seg_pcd_np, file_path):
    """
    Speichert eine erweiterte Punktwolke mit Instanz- und Semantiklabels in einer .ply-Datei.

    Args:
        inst_seg_pcd_np (numpy.ndarray): Die Punktwolke mit Positionen, Farben, semantischen und Instanz-Labels.
        file_path (str): Pfad, unter dem die .ply-Datei gespeichert werden soll.

    Returns:
        open3d.t.geometry.PointCloud: Die erweiterte Punktwolke mit Instanz- und Semantiklabels.
    """
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
    """
    Erstellt eine Farbtabelle für die Visualisierung der Instanzsegmentierungsergebnisse.

    Args:
        num_labels (int): Anzahl der verschiedenen Instanzlabels.

    Returns:
        numpy.ndarray: Array mit RGB-Werten für jede Instanz.
    """
    #max_label = labels.max().item()
    cmap = plt.get_cmap("tab20", num_labels) # "tab20", "viridis"
    colors = cmap(np.arange(num_labels))
    colors = (colors[:, :3] * 255).astype(np.uint8)
    #print(f"colors.shape: {colors.shape}")
    return colors

# create pointcloud to show instance segmentation results
def create_pointcloud_for_instance_seg_vis(inst_seg_pcd_np, colors):
    """
    Erzeugt eine Punktwolke für die Visualisierung von Instanzsegmentierungsergebnissen.

    Args:
        inst_seg_pcd_np (numpy.ndarray): Punktwolke mit Instanzlabels.
        colors (numpy.ndarray): Array mit Farben für jede Instanz.

    Returns:
        open3d.t.geometry.PointCloud: Die erweiterte Punktwolke zur Visualisierung.
    """
    pcd_extended = o3d.t.geometry.PointCloud() # erstellt eine leere PointCloud
    pcd_extended.point.positions = inst_seg_pcd_np[:, 0:3] # fuellt sie mit Punkten
    inst_labels = (inst_seg_pcd_np[:, 7]).astype(np.uint8)
    pcd_extended.point.colors = colors[inst_labels, :]
    return pcd_extended

# visualize instance segmentation results as a pointcloud
def vis_instance_seg_results(inst_label_counter, inst_seg_pcd_np):
    """
    Visualisiert die Instanzsegmentierungsergebnisse als Punktwolke.

    Args:
        inst_label_counter (int): Anzahl der verschiedenen Instanzen.
        inst_seg_pcd_np (numpy.ndarray): Punktwolke mit Instanzlabels.

    Returns:
        None
    """
    colors = create_colors_for_instance_seg_vis(inst_label_counter) # +2 because for noise (label -1) + index 0
    #pcd_np = inst_seg_pcd_np.copy()
    #pcd_np[pcd_np[:, 7].astype(int) == -1, 7] = inst_label_counter + 1
    pcd = create_pointcloud_for_instance_seg_vis(inst_seg_pcd_np, colors)
    visualize_pointcloud(pcd.to_legacy())

def apply_dbscan_to_semantic_category(semantically_filtered_pcd, eps, min_points):
    """
    Wendet DBSCAN-Clusteralgorithmus auf eine semantische Kategorie an, um Instanzen zu trennen.

    Args:
        semantically_filtered_pcd (numpy.ndarray): Gefilterte Punktwolke mit derselben semantischen Kategorie.
        eps (float): Maximaler Abstand zwischen zwei Punkten für denselben Cluster.
        min_points (int): Minimale Anzahl von Punkten für einen Cluster.

    Returns:
        numpy.ndarray: Array mit den Clusterlabels für jeden Punkt (-1 bedeutet Rauschen).
    """
    part_pcd =  o3d.t.geometry.PointCloud() # erstellt eine leere PointCloud
    part_pcd.point.positions = semantically_filtered_pcd[:, 0:3] # fuellt die pointcloud mit xyz-Daten
    labels = part_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True) # dbscan
    labels = labels.numpy()
    return labels

def update_instance_labels(labels, inst_label_noise, inst_label_counter):
    """
    Aktualisiert die Instanzlabels nach der DBSCAN-Clustering.

    Args:
        labels (numpy.ndarray): Die DBSCAN-Clusterlabels (-1 = Rauschen).
        inst_label_noise (int): Label für Rauschen.
        inst_label_counter (int): Zähler für die Instanzlabels.

    Returns:
        tuple: (aktualisierte Labels, aktualisierter Instanzlabelzähler)
    """
    labels[labels.astype(int) == -1] = inst_label_noise
    max_label = labels.max()
    labels_updated = labels.copy()
    for dbscan_cluster in range(1, max_label+1):
        # if inst_label_counter == 7:
        #     print(f"len(labels[labels.astype(int) == dbscan_cluster]): {np.sum(labels.astype(int) == dbscan_cluster)}")
        mask = labels.astype(int) == dbscan_cluster
        labels_updated[mask] = inst_label_counter
        #labels[labels.astype(int) == dbscan_cluster] = inst_label_counter
        inst_label_counter += 1
    #print(f"labels: {set(labels)}")
    return (labels_updated, inst_label_counter)

def inst_seg_with_dbscan(pcd_np, num_sem_cls=13, eps=0.4, min_points=10):
    """
    Führt Instanzsegmentierung mit DBSCAN für jede semantische Kategorie in einer Punktwolke durch.

    Args:
        pcd_np (numpy.ndarray): Punktwolke mit Positionen, Farben und semantischen Labels.
        num_sem_cls (int, optional): Anzahl der semantischen Klassen. Standard ist 13.
        eps (float, optional): Maximaler Abstand für DBSCAN. Standard ist 0.4.
        min_points (int, optional): Minimale Anzahl an Punkten für DBSCAN. Standard ist 10.

    Returns:
        tuple: (instanzsegmentierte Punktwolke, Zähler der Instanzlabels)
    """
    inst_label_counter = 1 # zaehlvariable fuer die instanzlabel
    inst_label_noise = 0
    inst_seg_pcd_np = np.empty((0, pcd_np.shape[1]+1)) # stores results

    for semantic_cls in range(num_sem_cls): # wir iterieren ueber jedes semantic_cls
        semantic_labels_column = pcd_np[:, 6]
        mask = semantic_labels_column.astype(int) == semantic_cls

        semantically_filtered_pcd = pcd_np[mask, :] # only pointcloud data with the same semantic semantic_cls

        labels = apply_dbscan_to_semantic_category(semantically_filtered_pcd, eps, min_points)
        #print(f"labels of dbscan: {labels}")
        if len(labels) == 0:
            print(f"not available semantic_cls is: {semantic_cls}")
        if len(labels) > 0: 
            # only for visualization of part instance segmented results
            #print(f"labels of dbscan: {labels}")
            result_without_updated_instance_labels = np.concatenate((semantically_filtered_pcd, np.reshape(labels, (len(labels), 1))), axis=1)
            pcd_np_without_updated_instance_labels = result_without_updated_instance_labels.copy()
            #inst_seg_column = pcd_np[:, 7].astype(int)
            pcd_np_without_updated_instance_labels[pcd_np_without_updated_instance_labels[:, 7].astype(int) == -1, 7] = labels.max() + 1
            #print(f"labels.max is {labels.max()+1} for semantic category {semantic_cls}")
            #vis_instance_seg_results(inst_label_counter=labels.max() + 2, inst_seg_pcd_np=pcd_np_without_updated_instance_labels) # Visualisierung Teilsegmentierungen
            # end of visualization
            #print(f"labels before: {labels}")
            labels = np.add(labels, 1)
            #print(f"labels after: {labels}")
            #labels[labels.astype(int) == -1] = inst_label_noise
            #print(f"labels.max(): {labels.max()}")
            #inst_label_counter += labels.max()
            labels, inst_label_counter = update_instance_labels(labels, inst_label_noise, inst_label_counter)
            result = np.concatenate((semantically_filtered_pcd, np.reshape(labels, (len(labels), 1))), axis=1)
            inst_seg_pcd_np = np.vstack((inst_seg_pcd_np, result))
    return (inst_seg_pcd_np, inst_label_counter)

def remove_noise_from_point_cloud(pcd_np, noise_label):
    inst_seg_pcd_np_without_noise = pcd_np[pcd_np[:, 7].astype(np.uint8) != noise_label, :]
    return inst_seg_pcd_np_without_noise

def remove_outlier_from_point_cloud(pcd_np, outlier_removal_threshold): 
    inst_ids = pcd_np[:, 7].astype(np.uint8)
    number_unique_instances = len(set(inst_ids.flatten()))
    #print(f"oid: {set(inst_ids.flatten())}")
    #print(f"number of oids after noise processing: {number_unique_instances}")
    num_removed_outliers = 0
    for inst_id in set(inst_ids.flatten()):
        mask = pcd_np[:, 7] == inst_id
        numb_points_instance = np.sum(mask)
        mask_invert = np.invert(mask)
        if numb_points_instance <= outlier_removal_threshold:
            print(f"inst_id of removed outlier: {inst_id}")
            pcd_np = pcd_np[mask_invert, :]
            num_removed_outliers += 1
    #print(f"num_removed_outliers: {num_removed_outliers}")
    inst_ids = pcd_np[:, 7].astype(np.uint8)
    number_unique_instances = len(set(inst_ids.flatten()))
    #print(f"inst_ids: {set(inst_ids.flatten())}")
    #print(f"number_unique_instances: {number_unique_instances}")
    return pcd_np

def renumber_instance_ids(pcd_np, start_inst_label=0):
    inst_ids = pcd_np[:, 7].astype(np.uint8)
    unique_inst_ids = set(inst_ids.flatten())
    number_unique_instances = len(unique_inst_ids)
    new_inst_label = start_inst_label
    #print(f"len unique_inst_ids: {len(unique_inst_ids)}")
    renumbered_pcd_np = pcd_np.copy()
    for inst_id in unique_inst_ids:
        mask = pcd_np[:, 7].astype(np.uint8) == inst_id
        renumbered_pcd_np[mask, 7] = new_inst_label
        new_inst_label += 1
    #print(f"renumbered instance ids: {set(renumbered_pcd_np[:, 7].astype(np.uint8).flatten())}")
    return renumbered_pcd_np


def postprocess_instance_segmentation(pcd_np, noise_label, outlier_removal_threshold):
    # noise removal
    pcd_np_without_noise = remove_noise_from_point_cloud(pcd_np, noise_label) # noise removal: alles, was dbscan als noise eingeordnet hat (label -1 -> 0)
    # outlier removal
    pcd_np_postprocessed = remove_outlier_from_point_cloud(pcd_np_without_noise, outlier_removal_threshold)
    pcd_np_postprocessed = renumber_instance_ids(pcd_np_postprocessed)
    return pcd_np_postprocessed

txt_dir_path = os.path.join(BASE_DIR, "log6/dump")
txt_file_name = "Area_6_lounge_1_pred.txt"
txt_file_path = os.path.join(txt_dir_path, txt_file_name)

pcd = create_pointcloud_from_txt_file(txt_file_path)
# visualize_pointcloud(pcd.to_legacy()) # Visualisierung der Pointcloud mit korrekten Farben
        
# load data from txt file in a numpy array and run instance segmentation
pcd_np = np.loadtxt(txt_file_path, usecols = (0, 1, 2, 3, 4, 5, 7))
inst_seg_pcd_np, inst_label_counter = inst_seg_with_dbscan(pcd_np, min_points=10, eps=0.4)

# store results as txt file
file_path = os.path.join(txt_dir_path, txt_file_name[:-4])
file_path_inst_seg_result = file_path + "_extended.txt"
np.savetxt(file_path_inst_seg_result, inst_seg_pcd_np) # wieder einkommentieren

vis_instance_seg_results(inst_label_counter, inst_seg_pcd_np)

# postprocessing: remove outlier und remove noise
# outlier removal: das Entfernen von Instanzen, die nur wenige PUnkte enthalten und visuell nicht als Objekt erkennbar sind
# Outlier Removal, da Gruppen von Punkten oder Punkte entfernt werden, die fuer die Instanzsegmentierung nicht als relevant betrachtet werden
# Instanzen mit wenigen Punkten sind dann halt nicht-vollständige Objekte, die beim 3dssg eher zu Verwirrung fuehren
# daher werden sie jetzt entfernt
# alles was von dbscan als noise (label -1) ausgegeben wird, wird entfernt -> hier im Code wurde label -1 auf label 0 gemapped
# dementsprechend muss label 0 entfernt werden

print(f"inst_label_counter: {inst_label_counter}")
# oid = inst_seg_pcd_np[:, 7].astype(np.uint8)
# number_oid = len(set(oid.flatten()))
#print(f"oid: {len(oid)}")
#print(f"number of oids: {number_oid}")

# Schritt 0: entferne noise und outliers, renumbere die Instanzlabels von 0 bis (#Instanzen - 1)
inst_seg_pcd_np_postprocessed = postprocess_instance_segmentation(inst_seg_pcd_np, noise_label=0, outlier_removal_threshold=200)

# inst_ids = inst_seg_pcd_np_postprocessed[:, 7].astype(np.uint8)
# number_unique_instances = len(set(inst_ids.flatten()))
# print(f"inst_ids: {set(inst_ids.flatten())}")
# print(f"number_unique_instances: {number_unique_instances}") 

# Schritt 1: Iteriere über alle Instanzen
#   Schritt 2: Filtere die Punkte, die zu einer Instanz gehoeren
#   Schritt 3: Anzahl der Punkte berechnen
#   Schritt 4: Wenn Anzahl Punkte < Soll-Wert, dann Entferne alle Punkte der Instanz
# Schritt 5: Instanz IDs nochmal neu bestimmen oder so lassen (je nachdem, womit man leichter arbeiten kann)

# Visualisierung der Postprocessed PointCloud

# Visualisierung basierend auf den einzelnen semantischen Klassen
## Beginn Visualisierung einzelner semantischer Klassen
# sem_labels = inst_seg_pcd_np_postprocessed[:, 6].astype(np.uint8)
# sem_labels = sorted(set(sem_labels.flatten()))

# for sem_label in sem_labels: 
#     part_pcd_np = inst_seg_pcd_np_postprocessed[inst_seg_pcd_np_postprocessed[:, 6]==sem_label, :]
#     renumbered_part_pcd_np = renumber_instance_ids(part_pcd_np, start_inst_label=0)
#     inst_ids = renumbered_part_pcd_np[:, 7].astype(np.uint8)
#     unique_inst_ids = set(inst_ids.flatten())
#     number_unique_instances = len(unique_inst_ids)
#     #print(f"unique_inst_ids: {unique_inst_ids}")
#     #print(f"number_unique_instances: {number_unique_instances}")
#     vis_instance_seg_results(inst_label_counter=number_unique_instances, inst_seg_pcd_np=renumbered_part_pcd_np)
## Ende Visualisierung semantischer Klassen

vis_instance_seg_results(inst_label_counter=len(set(inst_seg_pcd_np_postprocessed[:, 7].astype(np.uint8).flatten())), inst_seg_pcd_np=inst_seg_pcd_np_postprocessed)

# save pointcloud (instance segmentation + postprocessing)
file_path = os.path.join(txt_dir_path, txt_file_name[:-4])
file_path_inst_seg_result = file_path + "_extended_postprocessed.txt"
#np.savetxt(file_path_inst_seg_result, inst_seg_pcd_np_postprocessed)

# Visualisierung jedes einzelnen Objekts
# das Objekt bekomme eine Farbe X, alle anderen Punkte bekommen Farbe Y

# Schritt 0: Farbe auswahlen, Instanzen (Labels) ermitteln
#colors = create_colors_for_instance_seg_vis(num_labels=2)
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
color_instance = np.array([31, 119, 180]).astype(np.uint8) # [ 31 119 180] blauton
color_rest = np.array([171, 173, 161]).astype(np.uint8) # grau-gelber Ton    #[158 218 229]
colors = np.array([color_rest, color_instance])
instances = set(inst_seg_pcd_np_postprocessed[:, 7].astype(np.uint8).flatten())
# Schritt 1: gehe jedes Objekt durch
for instance_label in instances:
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
    visualize_pointcloud(pcd.to_legacy())
#   Schritt 7: Zeige auch das semantische Label (als kleine Orientierung)
# Ende Visualisierung jedes einzelnen Objekts


