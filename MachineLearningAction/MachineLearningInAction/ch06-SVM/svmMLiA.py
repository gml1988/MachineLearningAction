def loadDataSet(fileName):
    dataMatrix = []; labelMatrix = []
    fr = open(fileName)
    for line in fr.readlines():
        lineAttribute = line.strip().split('\t')
        dataMatrix.append([float(lineAttribute[0]), float(lineAttribute[1])])
        labelMatrix.append(float(lineAttribute[2]))
    return dataMatrix, labelMatrix