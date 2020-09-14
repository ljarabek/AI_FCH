# sick_images_path = "C:/Users/LeonE/Desktop/PREŠERNOVA/AI FCH/DICOM"
from os.path import join

root = "/media/leon/2tbssd/PRESERNOVA"  # skupna pot
images_path_sick = join(root, "AI FCH/DICOM_all4mm")  # pot do bolnih
images_path_healthy = join(root, "PREŠERNOVA_ZDRAVI")  # pot do zdravih
csv_path = join(root,"data_good.csv")  # pot do csv z metapodatki
indices = join(root, "data_good_index")  # datoteka z vrsticami z imeni stolpcev iz csv
save_folder = join(root, "saves")  # se ne uporablja

m_list_settings = {'encoding': ('histo_lokacija', ["DS", "DZ", "LS", "LZ", "healthy"]), 'wanted_shape_ct': (200, 200),
                   'cropping': ((39, 16), (100, 32), (100, 32))}
                                        #  drug element v tuple "encoding" so imena,
                                        #  ki so v histo_lokacija stolpcu v csv
                                        #  wanted shape - oblika koncne slike ki je vhod v model,
                                        #  cropping - izrez slike od kod do kod, na vsaki od treh koordinat

