# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:53:17 2019

@author: Augustin Leclerc
"""
import csv
import os
import numpy
import math
import sys
import struct

os.chdir(".")

#constants
EXPERIENCE_NAME = "D-RootSPH-Anisotropic2d"
CSV_PATH = "./examples/Experiments/" + EXPERIENCE_NAME + "/Stu-Aniso_out/data/"

""" Transforms a string of a number written in scientific form to an integer """
def strSciToInt(str) :
    liste = str.split("E")
    exp = float(liste[1].strip())
    coef = float(liste[0].strip())
    return coef * pow(10, exp)

def createMatrix(line) :
    xx = strSciToInt(line["Qfxx"])
    yy = strSciToInt(line["Qfyy"])
    zz = strSciToInt(line["Qfzz"])
    xy = strSciToInt(line["Qfxy"])
    yz = strSciToInt(line["Qfyz"])
    xz = strSciToInt(line["Qfxz"])
    mat = numpy.zeros((3, 3))
    mat[0][0] = xx
    mat[0][1] = xy
    mat[0][2] = xz
    mat[1][0] = xy
    mat[1][1] = yy
    mat[1][2] = yz
    mat[2][0] = xz
    mat[2][1] = yz
    mat[2][2] = zz
    return mat

def computeTransformedMatrix(matOrg) :
    eigenValues, passMatrix = numpy.linalg.eig(matOrg)
    newDiag = numpy.zeros((3, 3))
    i = 0
    for val in eigenValues :
        newDiag[i][i] = 1./math.sqrt(val)
        i += 1
    return numpy.matmul(passMatrix, numpy.matmul(newDiag, passMatrix.transpose()))
    

def extractCsv(name) :
    dict_list = []
    with open(name) as csvfile:
        # recovering of the variables names
        fields = ["Pos.x", "Pos.y", "Pos.z", "Idp", "Vel.x", "Vel.y", "Vel.z", "Rhop", "Mass", "Press", "Type", "Qfxx", "Qfxy", "Qfxz", "Qfyy", "Qfyz", "Qfzz"]
        reader = csv.DictReader(csvfile, fieldnames = fields, delimiter=";")
        for row in reader:
            dict_list.append(row)
        del dict_list[0:3]
    return dict_list
        
def createVtk(dic, name, folder, step) :
    fic = open(folder + "/" + name + "All_{:04d}.vtk".format(step), "w")
    fic.write("# vtk DataFile Version 3.0\n")
    fic.write("Try\n")
    fic.write("ASCII\n")
    fic.write("DATASET POLYDATA\n")
    
    fic.write("POINTS {} float\n".format(len(dic)))
    for line in dic :
        fic.write("{} {} {}\n".format(line["Pos.x"], line["Pos.y"], line["Pos.z"]))
    
    fic.write("VERTICES {} {}\n".format(len(dic), len(dic) * 2))
    i = 0
    for line in dic :
        fic.write("1 {}\n".format(i))
        i += 1
    
    fic.write("POINT_DATA {}\n".format(len(dic)))
    fic.write("SCALARS Idp unsigned_int\n")
    fic.write("LOOKUP_TABLE default\n")
    for line in dic :
        fic.write("{} ".format(line["Idp"]))
    fic.write("\n")
    
    fic.write("FIELD FieldData 5\n")
    
    fic.write("Vel 3 {} float\n".format(len(dic)))
    for line in dic :
        fic.write("{} {} {}".format(line["Vel.x"], line["Vel.y"], line["Vel.z"])) # a voir
    fic.write("\n")
    
    fic.write("Rhop 1 {} float\n".format(len(dic)))
    for line in dic :
        fic.write("{} ".format(line["Rhop"]))
    fic.write("\n")
    
    fic.write("Mass 1 {} float\n".format(len(dic)))
    for line in dic :
        fic.write("{} ".format(line["Mass"]))
    fic.write("\n")
    
    fic.write("Press 1 {} float\n".format(len(dic)))
    for line in dic :
        fic.write("{} ".format(line["Press"]))
    fic.write("\n")
    
    fic.write("Type 1 {} unsigned_char\n".format(len(dic)))
    for line in dic :
        fic.write("{} ".format(line["Type"]))
    fic.write("\n")
    
    fic.write("TENSORS tensors1 float\n")
    for line in dic :
        mat = createMatrix(line)
        newMat = computeTransformedMatrix(mat)
        fic.write("{} {} {}\n{} {} {}\n{} {} {}\n\n".format(newMat[0][0], newMat[0][1], newMat[0][2], newMat[1][0], newMat[1][1], newMat[1][2], newMat[2][0], newMat[2][1], newMat[2][2]))
    
arguments = sys.argv
name = arguments[1]
folder = arguments[2]
i = 0
while os.path.exists(folder + "/" + name + "All_{:04d}.csv".format(i)) :
    dic = extractCsv(folder + "/" + name + "All_{:04d}.csv".format(i))
    createVtk(dic, name, folder, i)
    print(name + "All_{:04d}.vtk created".format(i))
    i += 1