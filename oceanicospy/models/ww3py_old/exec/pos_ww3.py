import os,sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

import subprocess
import shutil
import util
import glob
import os

class files_from_data_dir():
    def __init__(self,name,root_path,spartan_path,ini_date):
        self.name = name
        self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
        self.spartan_path = spartan_path
        self.idate = ini_date

    def copy_files(self):
        self.out_path=f'{self.spartan_path}{self.idate.strftime("%Y%m%d%H")}_{self.name}/'
        if util.verify_files(f'{self.out_path}out_grd.ww3') and util.verify_files(f'{self.out_path}out_pnt.ww3'):
            util.verify_links('out_grd.ww3',self.out_path,self.run_path)
            util.verify_links('out_pnt.ww3',self.out_path,self.run_path)

            os.system(f'cp {self.spartan_path}{self.idate.strftime("%Y%m%d%H")}_{self.name}/log.ww3 {self.run_path}')
        else:
            raise UserWarning('WW3 has to be runned before post-processing the results')

class ounf():
    def __init__(self,root_path,ini_date,ounf_dict):
        self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
        self.inp_path = f'{root_path}info/inp/'
        self.ounf_dict = ounf_dict

    def fill_ounf(self):
        shutil.copy(f'{self.inp_path}ww3_ounf.inp_code', f'{self.run_path}ww3_ounf.inp')
        util.fill_files(f'{self.run_path}ww3_ounf.inp',self.ounf_dict)

    def run_ounf(self):
        print ('\n*** Running post-processing to generate field outputs ***\n')
        f = open(f'{self.run_path}ounf.log', "w") 
        subprocess.call([f'{self.run_path}ww3_ounf'],cwd=self.run_path,stdout=f)  

class ounp():
    def __init__(self,root_path,ini_date,ounp_dict):
        self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
        self.idate = ini_date
        self.inp_path = f'{root_path}info/inp/'
        self.ounp_dict = ounp_dict

    def fill_ounp(self,dicti):
        shutil.copy(f'{self.inp_path}ww3_ounp.inp_code', f'{self.run_path}ww3_ounp.inp')
        util.fill_files(f'{self.run_path}ww3_ounp.inp',dicti)

    def run_ounp(self):
        print ('\n*** Running post-processing to generate point outputs ***\n') 

        self.all_dicts=[]                

        for i in range (1,4):
            if i == 1:
                for j in range (2,4):
                    self.little_dic=dict(out_type=str(i),sub_spectra=str(j),sub_tab_params='$2',sub_src='$2')
                    self.all_dicts.append(self.little_dic)
            elif i ==2:
                for x in range(1,3,1):
                    self.little_dic=dict(out_type=str(i),sub_spectra='$3',sub_tab_params=str(x),sub_src='$2')
                    self.all_dicts.append(self.little_dic)
            else:
                for k in range (2,5,2):
                    self.little_dic=dict(out_type=str(i),sub_spectra='$3',sub_tab_params='$2',sub_src=str(k))
                    self.all_dicts.append(self.little_dic)
            
        for val in self.all_dicts:
            self.ounp_dict.update(val)
            self.fill_ounp(self.ounp_dict)

            f = open(f'{self.run_path}ounp.log', "w") 
            subprocess.call([f'{self.run_path}ww3_ounp'],cwd=self.run_path,stdout=f)  

            if self.ounp_dict['out_type']=='1' and self.ounp_dict['sub_spectra']=='2':
                os.system(f'mv {self.run_path}ww3.{self.idate.strftime("%Y")}_tab.nc {self.run_path}ww3.{self.idate.strftime("%Y")}_spec_1d.nc')

            elif self.ounp_dict['out_type']=='1' and self.ounp_dict['sub_spectra']=='3':
                os.system(f'mv {self.run_path}ww3.{self.idate.strftime("%Y")}_spec.nc {self.run_path}ww3.{self.idate.strftime("%Y")}_spec_2d.nc')

            elif self.ounp_dict['out_type']=='2' and self.ounp_dict['sub_tab_params']=='2':
                os.system(f'mv {self.run_path}ww3.{self.idate.strftime("%Y")}_tab.nc {self.run_path}ww3.{self.idate.strftime("%Y")}_tab_params.nc')

            elif self.ounp_dict['out_type']=='2' and self.ounp_dict['sub_tab_params']=='1':
                os.system(f'mv {self.run_path}ww3.{self.idate.strftime("%Y")}_tab.nc {self.run_path}ww3.{self.idate.strftime("%Y")}_extra_params.nc')

            elif self.ounp_dict['out_type']=='3' and self.ounp_dict['sub_src']=='2':
                os.system(f'mv {self.run_path}ww3.{self.idate.strftime("%Y")}_tab.nc {self.run_path}ww3.{self.idate.strftime("%Y")}_src_1d.nc')

            elif self.ounp_dict['out_type']=='3' and self.ounp_dict['sub_src']=='4':
                os.system(f'mv {self.run_path}ww3.{self.idate.strftime("%Y")}_src.nc {self.run_path}ww3.{self.idate.strftime("%Y")}_src_2d.nc')

class copy_results():
    def __init__(self,root_path,ini_date) -> None:
        self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
        self.data_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/'

    def copy(self):
        print ('\n*** Copying results to data folder ***\n')

        os.system(f'rm -rf {self.data_path}*.nc')

        for filename in glob.glob(f'{self.run_path}ww3.*.nc'):
            shutil.copy(filename, f'{self.data_path}')

        for filename in glob.glob(f'{self.run_path}ww3_*.inp'):
            shutil.copy(filename, f'{self.data_path}')

        for filename in glob.glob(f'{self.run_path}*'):
            if 'log' in filename:
                shutil.copy(filename, f'{self.data_path}')

if __name__ == '__main__':
    # print_a() is only executed when the module is run directly.
    print('executed directly')