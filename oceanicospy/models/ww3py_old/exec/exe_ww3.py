import os,sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

import subprocess
import shutil
import util
import wget
import fileinput
import glob

class buoys_matcher():
    def __init__(self,root_path,ini_date) -> None:
        self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
        self.inp_plots_path = f'{root_path}info/plots/'
        self.idate = ini_date
        self.url_sufixs = {'stdmet':'h','swden':'w','swr1':'j','swr2':'k','swdir':'d','swdir2':'i'}

    def dwnd_one_buoy(self,buoy_id):
            self.path_buoy = f'{self.inp_plots_path}{buoy_id}/'
            os.system(f'mkdir {self.path_buoy}')
            for suf,label in self.url_sufixs.items():
                name_buoy_file=f'{buoy_id}{label}{self.idate.strftime("%Y")}.txt.gz'
                url = f'https://www.ndbc.noaa.gov/data/historical/{suf}/{name_buoy_file}'
                try:
                    filename = wget.download(url,self.path_buoy)
                    os.system(f'gzip -d {self.path_buoy}{name_buoy_file}')
                except:
                    pass


    def dwnd_all_buoys(self):
        # name_buoys=['42056','42057','42058','42059','42060']
        name_buoys=['42001','42003','42041']
        for buoy in name_buoys:
            if len(glob.glob(f'{self.inp_plots_path}{buoy}/*.txt'))==6:
                print ('All buoy data is already downloaded')
            else:
                self.dwnd_one_buoy(buoy)

class running_ww3():
    def __init__(self,name,root_path,spartan_path,ini_date,exe_path,shel_dict,run_mode):
        self.name = name
        self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
        self.spartan_path = spartan_path
        self.inp_path = f'{root_path}info/inp/'
        self.exe_path = exe_path
        self.shel_dict = shel_dict
        self.run_mode = run_mode
        self.idate = ini_date
    
    def fill_shel(self):
        print ('\n*** Editing ww3_shel.inp ***\n')
        shutil.copy(f'{self.inp_path}ww3_shel.inp_code', f'{self.run_path}ww3_shel.inp')
        util.fill_files(f'{self.run_path}ww3_shel.inp',self.shel_dict)

    def run_shel(self):
        print ('\n*** Running main program: ww3_shel ***\n')
        if self.run_mode!='spartan':
            f = open('shel.log', "w") 
            subprocess.call([f'{self.run_path}ww3_shel'],cwd=self.run_path,stdout=f)  
            shutil.move('shel.log', f'{self.run_path}shel.log')
        else:
            shutil.copy(f'{self.inp_path}template.slurm', f'{self.run_path}{self.name}.slurm')
            with fileinput.FileInput(f'{self.run_path}{self.name}.slurm',inplace=True, backup='') as file:
                for line in file:
                    print(line.replace('name_job',self.name),end='')

            os.system(f'mkdir -p {self.spartan_path}{self.idate.strftime("%Y%m%d%H")}_{self.name}')
            os.system(f'cp -r {self.run_path}wind.ww3 {self.spartan_path}{self.idate.strftime("%Y%m%d%H")}_{self.name}')
            os.system(f'cp -r {self.run_path}current.ww3 {self.spartan_path}{self.idate.strftime("%Y%m%d%H")}_{self.name}')
            os.system(f'cp -r {self.run_path}mod_def.ww3 {self.spartan_path}{self.idate.strftime("%Y%m%d%H")}_{self.name}')
            os.system(f'cp -r {self.run_path}restart.ww3 {self.spartan_path}{self.idate.strftime("%Y%m%d%H")}_{self.name}')
            os.system(f'cp -r {self.run_path}ww3_shel.inp {self.spartan_path}{self.idate.strftime("%Y%m%d%H")}_{self.name}')
            os.system(f'ln -s {self.exe_path}ww3_shel {self.spartan_path}{self.idate.strftime("%Y%m%d%H")}_{self.name}')
            os.system(f'cp -r {self.run_path}*.slurm {self.spartan_path}{self.idate.strftime("%Y%m%d%H")}_{self.name}')
        self.out_grd_path=f'{self.spartan_path}{self.idate.strftime("%Y%m%d%H")}_{self.name}/out_grd.ww3'
        self.out_pnt_path=f'{self.spartan_path}{self.idate.strftime("%Y%m%d%H")}_{self.name}/out_pnt.ww3'
        if util.verify_files(self.out_grd_path) and util.verify_files(self.out_pnt_path):
            print('\n The case has been runned in Spartan \n')
        else:
            raise UserWarning('The slurm file has to be launched to the queue for the execution \n with 2 nodes and 16 tasks can take around 3.5 hours')