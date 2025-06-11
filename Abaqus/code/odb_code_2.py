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
            random_num = int(200000000 * random.random())  
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
        
        pickedNodes =(n1[917:918], n1[912:913], n1[901:902], n1[878:879], 
            n1[850:851], n1[804:805], n1[853:854], n1[880:881], )
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
        # Read displacement of particular Strtch Node : 429
        session.xyDataListFromField(odb=self.odb, outputPosition=NODAL, variable=(('U', 
            NODAL, ((COMPONENT, 'U2'), )), ), nodeLabels=(('PART-1-1', ('429', )), ))
        
        session.xyDataObjects.changeKey(fromName='ADD_RF:RF2', toName=RF_name)
        session.xyDataObjects.changeKey(fromName='U:U2 PI: PART-1-1 N: 429', 
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
        # print("Displacement array sample:", self.displacement[:5])
        # print("Force array sample:", self.force[:5])

    def save_data(self, paras_path):
        paras = np.genfromtxt(paras_path, delimiter=',', skip_header=1)
        id = self.index
        Equal_S = paras[id][6]
        Equal_L = paras[id][7]
        self.moduli = self.F_D *(Equal_L/Equal_S) 

        
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
        
        self.origin_1 = self.o.steps['Step-1'].frames[0].fieldOutputs['COORD'].getSubset(position=NODAL).values[429].data
        self.origin_2 = self.o.steps['Step-1'].frames[0].fieldOutputs['COORD'].getSubset(position=NODAL).values[169].data
        self.origin_3 = self.o.steps['Step-1'].frames[0].fieldOutputs['COORD'].getSubset(position=NODAL).values[881].data
        self.origin_4 = self.o.steps['Step-1'].frames[0].fieldOutputs['COORD'].getSubset(position=NODAL).values[705].data
        # print(self.origin_1)
        self.end_1 = self.o.steps['Step-1'].frames[-1].fieldOutputs['COORD'].getSubset(position=NODAL).values[429].data
        self.end_2 = self.o.steps['Step-1'].frames[-1].fieldOutputs['COORD'].getSubset(position=NODAL).values[169].data
        self.end_3 = self.o.steps['Step-1'].frames[-1].fieldOutputs['COORD'].getSubset(position=NODAL).values[881].data
        self.end_4 = self.o.steps['Step-1'].frames[-1].fieldOutputs['COORD'].getSubset(position=NODAL).values[705].data

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
        
    def odb_process(self):
        self.get_moduli()
        self.get_possion()
        print('in Job-{}, possion ratio is {}'.format(self.index, self.mu))
        print("Force-Displacement slope is : ", self.F_D)
        
    def close_odb(self):
        self.o.close()

if __name__ == "__main__":
    type_id = 3
    #odb_base_dirpath = 'F:/simulation_data/type{}/odb_files'.format(type_id)
    odb_base_dirpath = 'F:/multi_function/mirror_4chiral_2/odb_files'
    # cae_base_dirpath = 'F:/simulation_data/type{}/save_dics'.format(type_id)
    cae_base_dirpath = 'F:/multi_function/mirror_4chiral_2/save_dics'
    temporary_storage_path = 'F:/multi_function/mirror_4chiral_2/temporary_storage'
    
    txt_dir_path = 'E:/01_Graduate_projects/Cellular_structures/Multi-functional_design/Code_Project/Abaqus/txt_path'
    
    paras_path = 'E:/01_Graduate_projects/Cellular_structures/Multi-functional_design/Code_Project/Abaqus/paras2.csv'
    i = 0
    
    results = []
        
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
                odb_data.save_data(paras_path)
                results.append([odb_data.index, odb_data.moduli, odb_data.mu])

    if results:
        results_array = np.array(results)
        results_csv_path = os.path.join(txt_dir_path, 'results.csv')
        np.savetxt(
            results_csv_path,
            results_array,
            delimiter=',',
            header='Index,Moduli,Poisson Ratio',
            comments='',
            fmt='%.6f'  # 控制浮点数精度
        )
        print("ok")
    else:
        print("Error")