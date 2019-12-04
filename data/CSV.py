import csv
from constants import *
import io
from data.sumniki import popravi_sumnike


def get_csv():
    csv_ = list()
    with io.open(csv_path, newline="", encoding="utf-8") as f:  # utf encoding due to čšđžć
        reader = csv.reader(f, delimiter=" ", quotechar="|")

        for i, row in enumerate(reader):
            if i == 0:
                keys = row[0].split(";")
            else:
                csv_.append(row[0].split(";"))
    patients_ = dict()
    for i in range(110):
        patients_[str(i)] = dict()
    for ida, attr in enumerate(keys):
        attr = popravi_sumnike(attr)
        for patient in csv_:
            if ida > 13:  # csv le do "histo" opombe
                continue

            else:
                try:
                    if "ID" in attr:
                        attr = "ID"
                    patients_[patient[0]][attr] = popravi_sumnike(patient[ida])
                except:
                    print(patient)
    patients = dict()
    for p in patients_:
        if patients_[p] != dict():
            patients[p] = patients_[p]
    return patients

