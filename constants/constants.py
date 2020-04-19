# sick_images_path = "C:/Users/LeonE/Desktop/PREŠERNOVA/AI FCH/DICOM"
from os.path import join

root = "/media/leon/2tbssd/PRESERNOVA"
images_path_sick = join(root, "AI FCH/DICOM_all4mm")
images_path_healthy = join(root, "PREŠERNOVA_ZDRAVI")
csv_path = join(root,"data_good.csv")
indices = join(root, "data_good_index")
save_folder = join(root, "saves")

m_list_settings = {'encoding': ('histo_lokacija', ["DS", "DZ", "LS", "LZ", "healthy"]), 'wanted_shape_ct': (200, 200),
                   'cropping': ((39, 16), (100, 32), (100, 32))}  # keep from source +- delta: slices, x, y

