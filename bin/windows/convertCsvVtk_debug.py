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

os.chdir(".")

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
    eigenValues, passMatrix = numpy.linalg.eig(matOrg);
    newDiag = numpy.zeros((3, 3))
    i = 0
    for val in eigenValues :
        newDiag[i][i] = 1./math.sqrt(val)
        i += 1
    return numpy.matmul(passMatrix, numpy.matmul(newDiag, passMatrix.transpose()))
    

def extractCsv(name, fields) :
    dict_list = []
    with open(name) as csvfile:
        # recovering of the variables names
        # fields = ["Pos.x", "Pos.y", "Pos.z", "Idp", "Vel.x", "Vel.y", "Vel.z", "Rhop", "Mass", "Press", "Type", "Qfxx", "Qfxy", "Qfxz", "Qfyy", "Qfyz", "Qfzz"]
        reader = csv.DictReader(csvfile, fieldnames = fields, delimiter=";")
        for row in reader:
            dict_list.append(row)
        del dict_list[0:3]
    return dict_list
        
def writeVariable(fic, dic, field, nb, type_) :
    fic.write("{} {} {} {}\n".format(field, nb, len(dic), type_))
    for line in dic :
        if nb == 1 :
            fic.write("{} ".format(line[field]))
        elif nb == 3 :
            fic.write("{} {} {} ".format(line[field + ".x"], line[field + ".y"], line[field + ".z"]))
    fic.write("\n")

def createVtk(dic, name, folder, step, fields) :
    specialFields = ["Pos.x", "Pos.y", "Pos.z", "Idp", "Qfxx", "Qfxy", "Qfxz", "Qfyy", "Qfyz", "Qfzz", "Vel.y", "Vel.z"]
    
    fic = open(folder + "/" + name + "_{:04d}.vtk".format(step), "w")
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
    
    fielsVar = []
    for f in fields :
        if f not in specialFields :
            fielsVar.append(f)
    
    fic.write("FIELD FieldData {}\n".format(len(fielsVar)))
    
    for f in fielsVar :
        if f == "Vel.x" :
             writeVariable(fic, dic, "Vel", 3, "float")
        elif f == "Type" :
            writeVariable(fic, dic, "Type", 1, "unsigned_char")
        else :
            writeVariable(fic, dic, f, 1, "float")
    """writeVariable(fic, dic, "Vel", 3, "float")
    writeVariable(fic, dic, "Rhop", 1, "float")
    writeVariable(fic, dic, "Mass", 1, "float")
    writeVariable(fic, dic, "Press", 1, "float")
    writeVariable(fic, dic, "Type", 1, "unsigned_char")"""
    
    if "Qfxx" in fields :
        fic.write("TENSORS tensors1 float\n")
        for line in dic :	
            mat = createMatrix(line)
            newMat = computeTransformedMatrix(mat)
            fic.write("{} {} {}\n{} {} {}\n{} {} {}\n\n".format(newMat[0][0], newMat[0][1], newMat[0][2], newMat[1][0], newMat[1][1], newMat[1][2], newMat[2][0], newMat[2][1], newMat[2][2]))

arguments = sys.argv
name = arguments[1]
folder = arguments[2]
#fields = arguments[3].split(";")
fields = arguments[3:]
i = 0

while os.path.exists(folder + "/" + name + "_{:04d}.csv".format(i)) :
    dic = extractCsv(folder + "/" + name + "_{:04d}.csv".format(i), fields)
    createVtk(dic, name, folder, i, fields)
    print(name + "_{:04d}.vtk created".format(i))
    i += 1