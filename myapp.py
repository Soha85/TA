#!C:/Users/Soha.DESKTOP-B8142EO/AppData/Local/Programs/Python/Python37/python.exe

import sys
import warnings
from KEM import KEM

warnings.filterwarnings("ignore")


if __name__=="__main__":
    
    if(len(sys.argv)>2):
        n=len(sys.argv)
        lst=[]
        for i in range(n):
            if(i >= 2):
                tmp=map(str, sys.argv[i].strip('[]').split(','))
                lst.append(list(tmp))
        k=KEM(sys.argv[1],lst)
        print(k.Extract_and_match())
    else:
        k=KEM(sys.argv[1])
        print(k.Extractonly())
        
    
