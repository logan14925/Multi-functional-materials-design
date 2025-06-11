# -*- coding: UTF-8 -*-

from abaqus import *
from odbAccess import *
from abaqusConstants import *
import visualization
import numpy as np
import os
from caeModules import *
from driverUtils import executeOnCaeStartup
import shutil
import random
executeOnCaeStartup()
Mdb()

fasten_nodes_set = [1428, 1423, 1412, 1391, 1368, 1329, 1289, 1227, 1292, 1331]
stretch_nodes_set = [613, 527, 456, 365, 300, 225, 172, 113, 72, 32]


class ODB_DATA:
    _generated_random_nums = set()
    def __init__(self, odb_file_path, temporary_storage_path, txt_dir_path):
        """_summary_

        Args:
            file_path (str): 当前访问的odb文件的文件路径
            txt_path (str): 后处理数据的保存路径
        """
        self.file_path = odb_file_path
        self.temp_path = temporary_storage_path
        self.txt_path = txt_dir_path
        
        self.random_num = self._generate_unique_random_num()
        self._basic_info()
        
        self.origin_coordinate = np.array([])
        self.output_coordinate = np.array([])
        
        self.odb = None
        self.p = None
        self.o = None
        self.nodes = None
        
        self.possion_ratio = None
        self.keypoint_possion = None
        self.stiffness = None
        self.max_rf = None
        
        
        self.o = session.openOdb(name=self.file_path, readOnly=False)
        session.viewports['Viewport: 1'].setValues(displayedObject=self.o)
        session.viewports['Viewport: 1'].makeCurrent()

        
    def _generate_unique_random_num(self):
        while True:
            random_num = int(20000 * random.random())  
            if random_num not in ODB_DATA._generated_random_nums:
                ODB_DATA._generated_random_nums.add(random_num) 
                return random_num
            
    def __del__(self):
        self.close_odb()
        print('-----------------delete-----------------')
        
    def _basic_info(self):
        """读取文件名所代表的信息:job_number(parameter index), load_ratio
        """
        odb_filename = os.path.splitext(os.path.basename(self.file_path))[0]
        parts = odb_filename.split("-")
        self.index = int(float(parts[1]))

    def create_set(self):
        self.p = self.o.rootAssembly.instances['PART-1-1']
        n1 = self.p.nodes
        self.nodes = self.p.nodes
        
        pickedNodes =(n1[1226:1227], n1[1288:1289], n1[1291:1292], n1[1328:1329], 
            n1[1330:1331], n1[1367:1368], n1[1390:1391], n1[1411:1412], n1[1422:1423], 
                    n1[1427:1428], )
        self.fasten_name = 'Fasten_set{}'.format(self.random_num)
        self.o.rootAssembly.NodeSet(name=self.fasten_name, nodes=pickedNodes)
        
        self.odb = session.odbs[self.file_path]
    
    def extract_forcedata(self):
        U2_name = "U2_{}".format(self.random_num)
        RF_name = "RF_SUM{}".format(self.random_num)
        XY_name = 'RF-U{}'.format(self.random_num)
        session.xyDataListFromField(odb=self.odb, outputPosition=NODAL, variable=(('RF', 
            NODAL, ((COMPONENT, 'RF2'), )), ), operator=ADD, nodeSets=(self.fasten_name, 
            ))
        session.viewports['Viewport: 1'].view.setValues(session.views['Front'])
        self.odb = session.odbs[self.file_path]
        session.xyDataListFromField(odb=self.odb, outputPosition=NODAL, variable=(('U', 
            NODAL, ((COMPONENT, 'U2'), )), ), nodeLabels=(('PART-1-1', ('613', )), ))
        
        session.xyDataObjects.changeKey(fromName='ADD_RF:RF2', toName=RF_name)
        session.xyDataObjects.changeKey(fromName='U:U2 PI: PART-1-1 N: 613', 
            toName=U2_name)
        
        xy1 = session.xyDataObjects[U2_name]
        xy2 = session.xyDataObjects[RF_name]
        xy3 = combine(xy1, xy2)
        xy3.setValues(sourceDescription='combine ( {},{} )'.format(U2_name, RF_name))
        tmpName = xy3.name
        session.xyDataObjects.changeKey(tmpName, XY_name)
        x0 = session.xyDataObjects[XY_name]
        self.forcedata_temp_path = os.path.join(self.temp_path, 'Job-{}_forcedata.txt'.format(self.index))
        session.writeXYReport(fileName=self.forcedata_temp_path, appendMode=OFF, xyData=(x0, 
            ))
        
    def read_forcedata(self):
        data = np.genfromtxt(self.forcedata_temp_path,
                            skip_header=1,    
                            delimiter=None,   
                            dtype=np.float64,
                            filling_values=0) 
        self.displacement = data[:, 0]  
        self.force = abs(data[:, 1])         
        self.F_D = self.force[-1] / self.displacement[-1]
        print("Displacement array sample:", self.displacement[:5])
        print("Force array sample:", self.force[:5])
        print("Force-Displacement slope is : ", self.F_D)
        
    def get_moduli(self):
        self.create_set()
        self.extract_forcedata()
        self.read_forcedata()
    
    def extract_coordinates(self):
        # While extracting field output nodally, index usually starts from 0, indiccating that node index need to subtract 1
        # If node index is 1428, use values[1427]
        
        # 1 -> upper left
        # 2 -> upper right
        # 3 -> lower left
        # 4 -> lower right
        
        #   1--------2
        #   |        |
        #   |        |
        #   |        |
        #   3--------4
        
        self.origin_1 = self.o.steps['Step-1'].frames[0].fieldOutputs['COORD'].getSubset(position=NODAL).values[613].data
        self.origin_2 = self.o.steps['Step-1'].frames[0].fieldOutputs['COORD'].getSubset(position=NODAL).values[169].data
        self.origin_3 = self.o.steps['Step-1'].frames[0].fieldOutputs['COORD'].getSubset(position=NODAL).values[1391].data
        self.origin_4 = self.o.steps['Step-1'].frames[0].fieldOutputs['COORD'].getSubset(position=NODAL).values[1099].data
        # print(self.origin_1)
        self.end_1 = self.o.steps['Step-1'].frames[-1].fieldOutputs['COORD'].getSubset(position=NODAL).values[613].data
        self.end_2 = self.o.steps['Step-1'].frames[-1].fieldOutputs['COORD'].getSubset(position=NODAL).values[169].data
        self.end_3 = self.o.steps['Step-1'].frames[-1].fieldOutputs['COORD'].getSubset(position=NODAL).values[1391].data
        self.end_4 = self.o.steps['Step-1'].frames[-1].fieldOutputs['COORD'].getSubset(position=NODAL).values[1099].data

    def calc_possion(self):
        
        origin_x_length = self.origin_2[0] - self.origin_1[0]
        origin_y_length = self.origin_1[1] - self.origin_3[1]
        output_x_length = self.end_2[0] - self.end_1[0]
        output_y_length = self.end_1[1]- self.end_3[1]
        
        delta_x = output_x_length - origin_x_length
        delta_y = output_y_length - origin_y_length
        strain_x = delta_x / origin_x_length
        strain_y = delta_y / origin_y_length

        self.mu = - strain_x / strain_y
    
    def get_possion(self):
        self.extract_coordinates()
        self.calc_possion()
        
    def odb_process(self, mu):
        self.get_moduli()
        self.get_possion()
        print('in Job-{}, possion ratio is {}'.format(self.index, self.mu))
        mu.append(self.mu)
        print(mu)
    def close_odb(self):
        self.o.close()



class CAE_DATA:
    def __init__(self, cae_path):
        self.cae_path = cae_path
        self.weight = None
        
        self.mdb = openMdb(pathName=self.cae_path)
        session.viewports['Viewport: 1'].setValues(displayedObject=None)
        p = mdb.models['Model-1'].parts['Part-1']
        session.viewports['Viewport: 1'].setValues(displayedObject=p)
    def __del__(self):
        self.cae_close()
        print('delete')
        
    def get_mass(self):
        assembly = self.mdb.models['Model-1'].rootAssembly
        self.weight = assembly.getMassProperties()['mass']
        # print(self.weight)
        
    def cae_close(self):
        self.mdb.close()

if __name__ == "__main__":
    type_id = 3
    #odb_base_dirpath = 'F:/simulation_data/type{}/odb_files'.format(type_id)
    odb_base_dirpath = 'F:/multi_function/mirror_4chiral/odb_files'
    # cae_base_dirpath = 'F:/simulation_data/type{}/save_dics'.format(type_id)
    cae_base_dirpath = 'F:/multi_function/mirror_4chiral/save_dics'
    temporary_storage_path = 'F:/multi_function/mirror_4chiral/temporary_storage'
    
    txt_dir_path = 'E:/01_Graduate_projects/Cellular_structures/Multi-functional_design/Code_Project/Abaqus/txt_path'
    
    mu = []
    
    i = 0
    for root, dirs, files in os.walk(odb_base_dirpath):
        for filename in files:
            print(filename)
            if filename.endswith('.odb'):

                odb_file_path = os.path.join(root, filename)
                print('file_path ', odb_file_path)
                file_name = os.path.splitext(os.path.basename(odb_file_path))[0]
                # print('name: ', file_name)
                odb_data = ODB_DATA(odb_file_path, temporary_storage_path, txt_dir_path)
                odb_data.odb_process()
                mu.append(odb_data.mu)




