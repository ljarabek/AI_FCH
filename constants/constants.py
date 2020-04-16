# sick_images_path = "C:/Users/LeonE/Desktop/PREŠERNOVA/AI FCH/DICOM"
from os.path import join

root = "/media/leon/2tbssd/PRESERNOVA"
images_path_sick = join(root, "AI FCH/DICOM_all4mm")
images_path_healthy = join(root, "PREŠERNOVA_ZDRAVI")
#csv_path = join(root, "DATA_FULL.csv")
csv_path = join(root,"data_good.csv")
indices = join(root, "data_good_index")
save_folder = join(root, "saves")

x_0, y_0 = 256, 256
x_r, y_r = 128, 128

pet_mean, pet_std = 200.91218981809996, 433.57059788198114
ct_mean, ct_std = -702.7254494258335, 453.584221218243
legal_labels = "LS", "LZ", "DS"
#label_list = [legal_labels, 'normal']
label_list = ["LS", "LZ", "DS", 'normal']

"""def get_book_variable_module_name(module_name):
    module = globals().get(module_name, None)
    book = {}
    if module:
        book = {key: module.__dict__[key] for key in module.__dict__ if not (key.startswith('__') or key.startswith('_'))}
    return book


book = get_book_variable_module_name('constants')
for key in book:
    print(key, book[key])"""
