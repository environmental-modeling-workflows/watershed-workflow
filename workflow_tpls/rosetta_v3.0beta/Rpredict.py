'''
    Rosetta version 3-alpha (3a) 
    Pedotransfer functions by Schaap et al., 2001 and Zhang and Schaap, 2016.
    Copyright (C) 2016  Marcel G. Schaap

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    Marcel G. Schaap can be contacted at:
    mschaap@cals.arizona.edu

'''

#running this script:

# python Rpredic.py -h                                              # provides help
# python Rpredict.py --predict                                      # gets predictions with summary results  (MySQL DB implied)
# python Rpredict.py --predict  --sqlite=./sqlite/Rosetta.sqlite     # gets predictions with summary results  (sqlite, no MySQL)
# python Rpredict.py --predict  --raw                               # gets raw predictions (MySQL DB implied)


#tested on python 2.7
import numpy as N
import argparse
#import pandas
import time as T
from contextlib import closing
from ANN_Module import PTF_MODEL
from DB_Module import DB

parser = argparse.ArgumentParser(description='Rosetta 3 pedotransfer function interface example.')
parser.add_argument('--host', default='localhost', help='MySQl host')
parser.add_argument('--user', default='root', help='MySQl username')  # you want to change the username
parser.add_argument('--db', default='Rosetta', help='MySQl database')
parser.add_argument('--predict', action='store_true', help='get data and predict')
parser.add_argument('--check', action='store_true', help='check prediction (for development only)')
parser.add_argument('--raw', action='store_true', help='get raw data')
parser.add_argument('--sqlite', default='', help='sqlite path (ignores MySQL options)')  #using this option will ignore the mysql options.
parser.add_argument('-i', '--input', action='store', help='input from file ')
parser.add_argument('-o','--output', action='store', help='store predicted data')

args = parser.parse_args()
print(args.input)

data_in = N.genfromtxt(args.input, delimiter='',dtype=N.float).transpose()
print(data_in)
#data_in=data_in[0:3,]
#print(data_in)
#print(type(data_in))
# if header is read as data, use the following:
#data_in=data_in[1:]



def get_rosetta_cal_data(db,input_var):
# This is given as an example
# note that we use the input_var to get model specific data
# so this function returns properly formated input data!
    print("Getting data from  database")
    data_query_clause="SELECT " + ",".join(ptf_model.input_var) +" FROM `Data`"
    with closing(db.get_cursor()) as cursor:
        cursor.execute(data_query_clause)
        data=N.array(list(cursor))
        data=data.transpose()
    print("Getting data from database, done (%s s)" % (T.time()-T0))
    return(data)

if args.predict:
    # host, user and db_name need to be named or None, even when using sqlite.
    # the block below could be integrated in other code.
    with DB(host=args.host, user=args.user, db_name=args.db, sqlite_path=args.sqlite) as db:

        print("Getting models from database")
        T0=T.time()

        #IMPORTANT LINE!!!
        ptf_model=PTF_MODEL(3, db)
        #if predicting multiple times, keep the ptf_model object around, otherwise you'll be hitting the DB a lot.

        #MODELS (note model 1 and 101 are still missing: textural tables of parameters) 
        # 2 NEW SSC (sand, silt, clay)
        # 3 NEW SSC BD (sand, silt, clay, bulk density)
        # 4 NEW SSC BD TH33 (sand, silt, clay, bulk density, theta at 33 kPa)
        # 5 NEW SSC BD TH33 TH1500 (sand, silt, clay, bulk density, theta at 33 kPa and 1500 kPa)
        # 102 OLD SSC
        # 103 OLD SSC BD
        # 104 OLD SSC BD TH33
        # 105 OLD SSC BD TH33 TH1500

        # UNITS!
        # SSC in weight %
        # BD in g/cm3
        # TH33 and T1500 as cm3/cm3

        # OUTPUT
        # theta_r [cm3/cm3]
        # theta_s [cm3/cm3]
        # alpha  [1/cm]
        # n
        # Ks in [cm/day]
        # standard deviations apply to the log10 forms for alpha, n and KS
        # NOT their their antilog forms 

        print("Getting models from database, done (%s s)" % (T.time()-T0))

        #This is an EXAMPLE for getting data from a database
        # ptf_model.input_var gives a list of required input data
        # IN THE CORRECT ORDER!!!
        #data=get_rosetta_cal_data(db, ptf_model.input_var)
        # USER must figure out how to get his/her own data
        # Hard coded data example
        # So data must be offered as a NUMPY matrix with the shape (ninput_var, nsamp)  (this was a Matlab-ann convention)
        # Here we create a (nsamp,ninput) matrix that we transpose immediately to  (ninput,nsamp)
        # Warning: here we implicitly assume that we're using model 2 or 102 because we provide only sand, silt and clay %
        #data=N.array([[90,5,5],[1,1,98],[1,2,97],[2,3,95]],dtype=N.float).transpose()
        #data=N.array([[90,5,5,1.4],[1,1,98,1.0],[2,5,93,1.2]],dtype=N.float).transpose()
        #data=data_in.transpose()
        data=data_in
        print (data)
        #IMPORTANT: note transpose if a model has 3 input variables and predictions need
        # to be made for N samples, the data array must have the shape (3,N) 
        # NOT N,3.
        # this is due to a Matlab convention how input and output are dimensionalized
        # (and contrary to intuition!)

        T0=T.time()
        print("Processing")
        # Development - may be needed for detecting numerical errors
        #N.seterr(over='raise') # debug or new model

        if not args.raw:
            # At UA we've been playing with levi-distributions to characterize the shape of the estimated distributions
            # This is NOT ready for general use (and it may never be because the estimation of levi distributions is somewhat unstable)
            # Below we print means (which is what most people want), stdv (uncertainty), 
            # as well as skewness and kurtosis (which may NOT be reliable)
            # the covariance is also printed as a 3D matrix
            # shape of mean,stdev,skew,kurt: (nout,nsamp)
            # shape of cov: (nout,nout,nsamp)

            #IMPORTANT LINE!!!
            res_dict = ptf_model.predict(data,sum_data=True) 
            # with sum_data=False you get the raw output WITHOUT Summary statistics
            # res_dict['sum_res_mean'] output log10 of VG-alpha,VG-n, and Ks

            print("Processing done (%s s)" % (T.time()-T0))
            vgm_name=res_dict['var_names']
            print(vgm_name)
            vgm_mean=res_dict['sum_res_mean']
            vgm_new=N.stack((vgm_mean[0],vgm_mean[1],10**vgm_mean[2],10**vgm_mean[3],10**vgm_mean[4]))
            vgm_new=vgm_new.transpose()
            print(vgm_new)
            # output estimation
            N.savetxt(args.output, vgm_new, delimiter=',',fmt='%f')  

            #print("STDV",)
            #print(res_dict['sum_res_std'])
            #print("SKEW")
            #print(res_dict['sum_res_skew'])
            #print("KURT")
            #print(res_dict['sum_res_kurt'])
            #print("COV")
            # Note we print the FIRST sample ONLY to prevent clutter. Normally you would not want this (so replace the 0 with :)
            #print(res_dict['sum_res_cov'][:,:,0])

        else:
            # we requested raw input.  You'll need to process these yourself.

            res_dict = ptf_model.predict(data,sum_data=False)  
            #shape: (nboot,nout,nsamp)
            #Warning: old rosetta (models 102..105) can ONLY provide retention parameters (not Ks)
            #New models (2..5) provide retention+ks
            #This is because the retention and ks models are synchronized in the new models wheras they were calibrated on different datasets in the old model
            # nboot is 1000, output is log10 of VG-alpha,VG-n, and Ks
            #print(res_dict['res'][0])
            #print(N.shape(res_dict['res'][0]))



if args.check:
# Note this is a development/debugging piece of code.
# However, given here as an example
# note that the different models require different input.
    with DB(host=args.host, user=args.user, db_name=args.db, sqlite_path=args.sqlite) as db:

        #model 102 (ssc, old rosetta)
        ptf_model=PTF_MODEL(102, db)
        data=N.array([[90,5,5],[40,35,25]],dtype=N.float).transpose()
        res_dict = ptf_model.predict(data,sum_data=True)  
        print(res_dict['var_names'])
        print("MEAN")
        print(res_dict['sum_res_mean'])

        #model 103 (sscbd, old rosetta)
        ptf_model=PTF_MODEL(103, db)
        data=N.array([[90,5,5,1.5],[40,35,25,1.5]],dtype=N.float).transpose()
        res_dict = ptf_model.predict(data,sum_data=True)  
        print(res_dict['var_names'])
        print("MEAN")
        print(res_dict['sum_res_mean'])

        #model 104 (sscbdth33, old rosetta)
        ptf_model=PTF_MODEL(104, db)
        data=N.array([[90,5,5,1.5,0.1],[40,35,25,1.5,0.2]],dtype=N.float).transpose()
        res_dict = ptf_model.predict(data,sum_data=True)  
        print(res_dict['var_names'])
        print("MEAN")
        print(res_dict['sum_res_mean'])

        #model 105 (sscbdth33th1500, old rosetta)
        ptf_model=PTF_MODEL(105, db)
        data=N.array([[90,5,5,1.5,0.1,0.05],[40,35,25,1.5,0.2,0.05]],dtype=N.float).transpose()
        res_dict = ptf_model.predict(data,sum_data=True)  
        print(res_dict['var_names'])
        print("MEAN")
        print(res_dict['sum_res_mean'])
