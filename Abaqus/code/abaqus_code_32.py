import numpy as np
from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
import json
import shutil

config_path = 'E:/01_Graduate_projects/Cellular_structures/Multi-functional_design/Code_Project/Abaqus/FEM_config.json'
with open(config_path, 'r') as file:
    config_params = json.load(file)
extrusion_height = config_params["extrusion_height"]
seed_size = config_params["seed_size"]
x_array = config_params["x_array"]
y_array = config_params["y_array"]
density = config_params["density"]
moduli = config_params["moduli"]
mu = config_params["mu"]

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  
        os.makedirs(path) 
        print("---{} created ---".format(path)) 
    else:
        print("---  There is this folder!  ---")

def _cae(paras, index, dic_path, odb_path):
    
    #################### init ###################
    session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=343.149230957031, 
        height=182.699996948242)
    session.viewports['Viewport: 1'].makeCurrent()
    session.viewports['Viewport: 1'].maximize()
    executeOnCaeStartup()
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=ON)
    Mdb()
    #: A new model database has been created.
    #: The model "Model-1" has been created.

    session.viewports['Viewport: 1'].setValues(displayedObject=None)
    os.chdir(dic_path)
    
    ################### part ##################
    thickness = paras[index][1]
    radius = paras[index][2]
    length_x = paras[index][3]
    length_y = paras[index][4]
    d1x = paras[index][5]
    d2x = paras[index][-2]
    d2y = paras[index][-1]

    s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
        sheetSize=200.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.Line(point1=(0.0, 20.0), point2=(0.0, 0.0))
    s.VerticalConstraint(entity=g[2], addUndoState=False)
    s.Line(point1=(0.0, 0.0), point2=(20.0, 0.0))
    s.HorizontalConstraint(entity=g[3], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[2], entity2=g[3], addUndoState=False)
    s.setAsConstruction(objectList=(g[2], g[3]))
    s.CircleByCenterPerimeter(center=(length_x, length_y), point1=(length_x+radius, length_y))
    s.DistanceDimension(entity1=v[3], entity2=g[2], textPoint=(5.75212097167969, 
        18.6266059875488), value=length_x)
    s.DistanceDimension(entity1=v[3], entity2=g[3], textPoint=(21.3918838500977, 
        3.15450286865234), value=length_y)
    
    s.Line(point1=(length_x, length_y+radius), point2=(0.0, length_y+radius))
    s.HorizontalConstraint(entity=g[5], addUndoState=False)
    s.TangentConstraint(entity1=g[4], entity2=g[5], addUndoState=False)
    s.CoincidentConstraint(entity1=v[5], entity2=g[4], addUndoState=False)
    s.CoincidentConstraint(entity1=v[6], entity2=g[2], addUndoState=False)
    s.Line(point1=(d1x, 0.0), point2=(length_x-radius, 10.0))
    s.CoincidentConstraint(entity1=v[7], entity2=g[3], addUndoState=False)
    s.CoincidentConstraint(entity1=v[8], entity2=g[4], addUndoState=False)
    s.HorizontalDimension(vertex1=v[7], vertex2=v[1], textPoint=(1.16546630859375, 
        -2.1781120300293), value=d1x)
    s.TangentConstraint(entity1=g[6], entity2=g[4])
    s.radialPattern(geomList=(g[5], ), vertexList=(), number=2, totalAngle=360.0, 
        centerPoint=(length_x, length_y))
    s.radialPattern(geomList=(g[6], ), vertexList=(), number=2, totalAngle=360.0, 
        centerPoint=(length_x, length_y))
    s.copyMirror(mirrorLine=g[2], objectList=(g[4], g[5], g[6], g[7], g[8]))
    s.copyMirror(mirrorLine=g[3], objectList=(g[2], g[4], g[5], g[6], g[7], g[8], 
        g[9], g[10], g[11], g[12], g[13]))
    s.linearPattern(geomList=(g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9], 
        g[10], g[11], g[12], g[13], g[14], g[15], g[16], g[17], g[18], g[19], 
        g[20], g[21], g[22], g[23], g[24]), vertexList=(), number1=x_array, 
        spacing1=4*length_x, angle1=0.0, number2=y_array, spacing2=4*length_y, angle2=90.0)
    p = mdb.models['Model-1'].Part(name='Part-1', dimensionality=TWO_D_PLANAR, 
        type=DEFORMABLE_BODY)
    p = mdb.models['Model-1'].parts['Part-1']
    p.BaseWire(sketch=s)
    s.unsetPrimaryObject()
    p = mdb.models['Model-1'].parts['Part-1']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models['Model-1'].sketches['__profile__']

    ############### Property  ########## 
    session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
        engineeringFeatures=ON)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=OFF)
    mdb.models['Model-1'].Material(name='Material-1')
    mdb.models['Model-1'].materials['Material-1'].Density(table=((density, ), ))
    mdb.models['Model-1'].materials['Material-1'].Elastic(table=((moduli, mu), ))
    mdb.models['Model-1'].RectangularProfile(name='Profile-1', a=extrusion_height, b=thickness)
    mdb.models['Model-1'].BeamSection(name='Section-1', 
        integration=DURING_ANALYSIS, poissonRatio=0.0, profile='Profile-1', 
        material='Material-1', temperatureVar=LINEAR, consistentMassMatrix=False)
    p = mdb.models['Model-1'].parts['Part-1']
    e = p.edges
    edges = e.getSequenceFromMask(mask=('[#ffffffff:18 ]', ), )
    region = p.Set(edges=edges, name='Set-1')
    p = mdb.models['Model-1'].parts['Part-1']
    p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
    p = mdb.models['Model-1'].parts['Part-1']
    e = p.edges
    edges = e.getSequenceFromMask(mask=('[#ffffffff:18 ]', ), )
    region=p.Set(edges=edges, name='Set-2')
    p = mdb.models['Model-1'].parts['Part-1']
    p.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(0.0, 0.0, 
        -1.0))

    ################ Assembly  ########## 
    a = mdb.models['Model-1'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
    a = mdb.models['Model-1'].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    p = mdb.models['Model-1'].parts['Part-1']
    a.Instance(name='Part-1-1', part=p, dependent=ON)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        adaptiveMeshConstraints=ON)
    
    ################ Step  ########## 
    mdb.models['Model-1'].StaticStep(name='Step-1', previous='Initial', 
        maxNumInc=1000, initialInc=0.005, maxInc=0.1)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
    mdb.models['Model-1'].steps['Step-1'].setValues(nlgeom=ON)
    mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
        'S', 'PE', 'PEEQ', 'PEMAG', 'LE', 'U', 'RF', 'CF', 'CSTRESS', 'CDISP', 
        'SVOL', 'EVOL', 'ESOL', 'IVOL', 'STH', 'COORD'))
    
    ################ Interaction  ########## 
        
    # ################ Load  ########## 
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
        predefinedFields=ON, connectors=ON, adaptiveMeshConstraints=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
    a = mdb.models['Model-1'].rootAssembly
    v1 = a.instances['Part-1-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=(
        '[#0:9 #100000 #0 #18 #20000 #80000200 #8200 ]', ), )
    region = a.Set(vertices=verts1, name='Set-1')
    
    # a = mdb.models['Model-1'].rootAssembly
    # instance = a.instances['Part-1-1']
    # new_nodes = instance.nodes.getByBoundingBox(-2*length_x*2-0.5, -100, 0, -2*length_x*2+0.5, +100, 0)
    # Fasten = a.Set(nodes=new_nodes, name='Fasten') 
    mdb.models['Model-1'].EncastreBC(name='BC-1', createStepName='Initial', 
        region=region, localCsys=None)
    
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
    a = mdb.models['Model-1'].rootAssembly
    v1 = a.instances['Part-1-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=(
        '[#8000000 #0 #20 #4000 #0 #200 #40000000', 
        ' #0:2 #200 #40000000 #0 #4000 ]', ), )
    region = a.Set(vertices=verts1, name='Set-2')
    
    # a = mdb.models['Model-1'].rootAssembly
    # instance = a.instances['Part-1-1']
    # new_nodes = instance.nodes.getByBoundingBox(3.5*x_array*4*length_x-0.5, -100, 0, 3.5*x_array*4*length_x+0.5, +100, 0)
    # Stretch = a.Set(nodes=new_nodes, name='Fasten') 
    mdb.models['Model-1'].DisplacementBC(name='BC-2', createStepName='Step-1', 
        region=region, u1=0.1 * length_x*4*x_array, u2=0.0, ur3=0.0, amplitude=UNSET, fixed=OFF, 
        distributionType=UNIFORM, fieldName='', localCsys=None)


    ################ Mesh  ########## 
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON, loads=OFF, 
        bcs=OFF, predefinedFields=OFF, connectors=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=ON)
    p = mdb.models['Model-1'].parts['Part-1']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, 
        engineeringFeatures=OFF, mesh=ON)
    session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(
        meshTechnique=ON)
    p = mdb.models['Model-1'].parts['Part-1']
    p.seedPart(size=seed_size, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models['Model-1'].parts['Part-1']
    p.generateMesh()
    a1 = mdb.models['Model-1'].rootAssembly
    a1.regenerate()
    a = mdb.models['Model-1'].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=OFF)
    
    ################ Job  ########## 
    mdb.Job(name='Job-{}'.format(index), model='Model-1', description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', resultsFormat=ODB, numThreadsPerMpiProcess=1, 
        multiprocessingMode=DEFAULT, numCpus=8, numDomains=8, numGPUs=1)
    mdb.jobs['Job-{}'.format(index)].submit(consistencyChecking=OFF)
    mdb.jobs['Job-{}'.format(index)].waitForCompletion()
    
    cae_path = os.path.join(dic_path, 'cae_{}'.format(index))
    mdb.saveAs(pathName=cae_path)
    # save cae file
    origin_odb_path = os.path.join(dic_path, 'Job-{}.odb'.format(index))
    # o3 = session.openOdb(name=origin_odb_path)
    target_odb_path = os.path.join(odb_path, 'Job-{}.odb'.format(index))
    shutil.copyfile(origin_odb_path, target_odb_path)

if __name__ == '__main__':
    all_save_path = 'F:/multi_function/mirror_4chiral_3'
    save_dics = os.path.join(all_save_path, 'save_dics')
    all_odb_path = os.path.join(all_save_path, 'odb_files')
    mkdir(all_odb_path)
    mkdir(save_dics)
    paras_path = 'E:/01_Graduate_projects/Cellular_structures/Multi-functional_design/Code_Project/Abaqus/paras3.csv'
    data = np.genfromtxt(paras_path, delimiter=',', skip_header=1)
    samples_num, _ = np.shape(data)
    
    for index in range(samples_num):
        dic_path = os.path.join(save_dics, 'Job-{}'.format(index))
        mkdir(dic_path)
        _cae(data, index, dic_path=dic_path, odb_path=all_odb_path)
