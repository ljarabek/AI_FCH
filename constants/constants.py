# sick_images_path = "C:/Users/LeonE/Desktop/PREŠERNOVA/AI FCH/DICOM"
from os.path import join

root = "/media/leon/2tbssd/PRESERNOVA"  # skupna pot
images_path_sick = join(root, "AI FCH/DICOM_all4mm")  # pot do bolnih
images_path_healthy = join(root, "PREŠERNOVA_ZDRAVI")  # pot do zdravih
csv_path = join(root, "data_good.csv")  # pot do csv z metapodatki
indices = join(root, "data_good_index")  # datoteka z vrsticami z imeni stolpcev iz csv
save_folder = join(root, "saves")  # se uporablja pri evalvaciji unceirtanty modela

m_list_settings = {'encoding': ('histo_lokacija', ["DS", "DZ", "LS", "LZ", "healthy"]), 'wanted_shape_ct': (200, 200),
                   'cropping': ((39, 16), (100, 32), (100, 32))}
#  drug element v tuple "encoding" so imena,
#  ki so v histo_lokacija stolpcu v csv
#  wanted shape CT - oblika PET, na katero se interpolira CT,
#  cropping - izrez slike od kod do kod, na vsaki od treh koordinat

master_pkl_dir = "/media/leon/2tbssd/PRESERNOVA/AI_FCH/data/master.pkl"

# def init_gc():
#    global global_count
#    global_count=0
# init_gc()
global_count = 0
