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
    thickness = paras[index][0]
    radius = paras[index][1]
    length = paras[index][2]
    arc = paras[index][3]

    
    s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
        sheetSize=200.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.Line(point1=(0.0, 25.0), point2=(0.0, 0.0))
    s.VerticalConstraint(entity=g[2], addUndoState=False)
    s.Line(point1=(0.0, 0.0), point2=(25.0, 0.0))
    s.HorizontalConstraint(entity=g[3], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[2], entity2=g[3], addUndoState=False)
    s.setAsConstruction(objectList=(g[2], ))
    s.setAsConstruction(objectList=(g[3], ))

    s.CircleByCenterPerimeter(center=(length/2, length/2), point1=(length/2+radius, length/2))
    session.viewports['Viewport: 1'].view.setValues(nearPlane=185.236, 
        farPlane=191.888, width=39.3175, height=15.855, cameraPosition=(-0.110476, 
        0.457746, 188.562), cameraTarget=(-0.110476, 0.457746, 0))
    
    tan = (length/2) / (length/2 - arc)
    theta = np.arctan(tan)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    arc_start_x = length/2 + radius * sin_theta
    arc_start_y = length/2 + radius * cos_theta
    arc_radius = ((length/2)**2 + (length/2 - arc)**2)**0.5
    arc_end_y = arc+arc_radius+radius
    
    s.ArcByCenterEnds(center=(0.0, arc), point1=(arc_start_x, arc_start_y), point2=(
        0.0, arc_end_y), direction=COUNTERCLOCKWISE)
    s.CoincidentConstraint(entity1=v[5], entity2=g[4], addUndoState=False)
    s.CoincidentConstraint(entity1=v[6], entity2=g[2], addUndoState=False)
    s.radialPattern(geomList=(g[5], ), vertexList=(), number=4, totalAngle=360.0, 
        centerPoint=(length/2, length/2))
    s.copyMirror(mirrorLine=g[2], objectList=(g[5], g[4], g[6], g[7], g[8]))
    s.copyMirror(mirrorLine=g[3], objectList=(g[9], g[10], g[11], g[12], g[13], 
        g[5], g[8], g[4], g[6], g[7]))
    s.linearPattern(geomList=(g[2], g[3], g[4], g[5], g[6], g[7], g[8], g[9], 
        g[10], g[11], g[12], g[13], g[14], g[15], g[16], g[17], g[18], g[19], 
        g[20], g[21], g[22], g[23]), vertexList=(), number1=5, spacing1=2*length, 
        angle1=0.0, number2=10, spacing2=2*length, angle2=90.0)
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

    mdb.models['Model-1'].materials['Material-1'].Density(table=((1.24e-09, ), ))
    mdb.models['Model-1'].materials['Material-1'].Elastic(table=((1690.0 * 7, 0.25), ))
    mdb.models['Model-1'].RectangularProfile(name='Profile-1', a=extrusion_height, b=thickness)
    mdb.models['Model-1'].BeamSection(name='Beam', integration=DURING_ANALYSIS, 
        poissonRatio=0.0, profile='Profile-1', material='Material-1', 
        temperatureVar=LINEAR, consistentMassMatrix=False)
    p = mdb.models['Model-1'].parts['Part-1']
    e = p.edges
    edges = e.getSequenceFromMask(mask=('[#ffffffff:106 #ff ]', ), )
    region = p.Set(edges=edges, name='Set-1')
    p = mdb.models['Model-1'].parts['Part-1']
    p.SectionAssignment(region=region, sectionName='Beam', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
    p = mdb.models['Model-1'].parts['Part-1']
    e = p.edges
    edges = e.getSequenceFromMask(mask=('[#ffffffff:106 #ff ]', ), )
    region=regionToolset.Region(edges=edges)
    p = mdb.models['Model-1'].parts['Part-1']
    p.assignBeamSectionOrientation(region=region, method=N1_COSINES, n1=(0.0, 0.0, 
        -1.0))

    ################ Assembly  ########## 
    #: Beam orientations have been assigned to the selected regions.
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
        maxNumInc=10000, initialInc=0.001, minInc=1e-08, maxInc=0.01)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
    mdb.models['Model-1'].steps['Step-1'].setValues(nlgeom=ON)
    mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
        'S', 'PE', 'PEEQ', 'PEMAG', 'LE', 'U', 'RF', 'CF', 'CSTRESS', 'CDISP', 
        'SVOL', 'EVOL', 'ESOL', 'IVOL', 'STH', 'COORD'))
    
    # ################ Load  ########## 
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
        predefinedFields=ON, connectors=ON, adaptiveMeshConstraints=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
    a = mdb.models['Model-1'].rootAssembly
    v1 = a.instances['Part-1-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=(
        '[#0:38 #400 #0 #900 #50000 #800000 #4000', ' #84008 ]', ), )
    region = a.Set(vertices=verts1, name='Set-1')
    mdb.models['Model-1'].EncastreBC(name='BC-1', createStepName='Initial', 
        region=region, localCsys=None)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
    a = mdb.models['Model-1'].rootAssembly
    v1 = a.instances['Part-1-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#80000000 #0 #80 #10000 #0 #800 #0', 
        ' #1 #0 #800 #0 #1000 #0:2 #80', ' #0 #4000 #0:2 #10 ]', ), )
    region = a.Set(vertices=verts1, name='Set-2')
    mdb.models['Model-1'].DisplacementBC(name='Stretch', createStepName='Step-1', 
        region=region, u1=0.0, u2= 20 * length * 0.1, ur3=UNSET, amplitude=UNSET, fixed=OFF, 
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
    session.viewports['Viewport: 1'].view.setValues(nearPlane=163.917, 
        farPlane=193.854, width=198.926, height=80.3746, viewOffsetX=-18.456, 
        viewOffsetY=-23.9929)
    session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
    session.viewports['Viewport: 1'].view.setValues(width=219.1, height=88.5258, 
        viewOffsetX=-2.34727, viewOffsetY=1.56524)
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
    all_save_path = 'F:/multi_function/mirror_4chiral'
    save_dics = os.path.join(all_save_path, 'save_dics')
    all_odb_path = os.path.join(all_save_path, 'odb_files')
    mkdir(all_odb_path)
    mkdir(save_dics)
    paras_path = 'E:/01_Graduate_projects/Cellular_structures/Multi-functional_design/Code_Project/Abaqus/paras.csv'
    data = np.genfromtxt(paras_path, delimiter=',', skip_header=1)
    samples_num, _ = np.shape(data)
    
    for index in range(samples_num):
        dic_path = os.path.join(save_dics, 'Job-{}'.format(index))
        mkdir(dic_path)
        _cae(data, index, dic_path=dic_path, odb_path=all_odb_path)