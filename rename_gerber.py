__author__ = 'msei'
import sys
import os

# maybe: http://daycounter.com/LabBook/Protel-Layer-File-Extensions.phtml

directory = "/Volumes/HaleBopp/Code/Projects/M3DCube/PCBs/IntelliBoard/"


def new_name(s):
    original_name = s
    s = s.replace(" - Bottom Copper (Resist).gbr", ".gbs")
    s = s.replace(" - Bottom Copper.gbr", ".gbl")
    s = s.replace(" - Bottom Silkscreen.gbr", ".gbo")
    #  file = file.replace("Drill Data - [Through Hole] (Unplated).drl", "")
    s = s.replace(" - Drill Data - [Through Hole].drl", ".dri")
    s = s.replace(" - Keepout.gbr", ".gko")
    s = s.replace(" - Top Copper (Paste).gbr", ".gtp")
    s = s.replace(" - Top Copper (Resist).gbr", ".gts")
    s = s.replace(" - Top Copper.gbr", ".gtl")
    s = s.replace(" - Top Silkscreen.gbr", ".gto")

    return s


for (dirpath, dirnames, filenames) in os.walk(directory):
    for filename in filenames:
        n = new_name(filename)

        if n != filename:
            print filename, "->", n
            os.rename(dirpath + "/" + filename, dirpath + "/" + n)
