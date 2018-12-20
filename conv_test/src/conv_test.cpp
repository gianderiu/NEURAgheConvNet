/******************************************************************************
 *                                                                            *
 *                   EOLAB @ DIEE - University of Cagliari                    *
 *                          Via Marengo, 2, 09123                             *
 *                       Cagliari - phone 070 675 5009                        *
 *                                                                            *
 * Author:       Gianfranco Deriu - gian.deriu@gmail.com                      *
 *                                                                            *
 * Project:     NEURAGHE - Accelerator for Convolutional neural network       *
 * File:                                                                      *
 * Description:                                                               *
 *                                                                            *
 ******************************************************************************/
#include "conv_test.h"

#include "assert.h"

#include <sys/time.h>
#include <time.h>



SOCMAP socs[2];
DATA* wPointer;


SPATCONV conv_params_hw;
SPATCONV conv_params_sw;
int n_weights;




void cnnMainInit(VARNAME load_data_dir)
{
	unsigned int o,i,j;
	printf("Init!\n");

	printf("\nIF:%d OF:%d max_OG:%d\n\n",_IF_,_OF_,_MAX_OG_);
	
	VARNAME filename;
	
	conv_params_hw = spatconv_create();           

	conv_params_hw->pout = (_OF_%_N_ROW_) ? _OF_+ _N_ROW_-_OF_%_N_ROW_: _OF_;
	if (_FS_==3)
		conv_params_hw->pin       = (_IF_%(_N_COL_*3)) ? _IF_+ _N_COL_*3-_IF_%(_N_COL_*3): _IF_;
	else
		conv_params_hw->pin       = (_IF_%(_N_COL_)) ? _IF_+ _N_COL_-_IF_%(_N_COL_) : _IF_;

	conv_params_hw->kern_s[0] = conv_params_hw->pout;
	conv_params_hw->kern_s[1] = conv_params_hw->pin;
	conv_params_hw->kern_s[2] = _FS_;
	conv_params_hw->kern_s[3] = _FS_;
	conv_params_hw->maxog     = _MAX_OG_;


	conv_params_sw = spatconv_create();

	conv_params_sw->pout      = _OF_;
	conv_params_sw->pin       = _IF_;
	conv_params_sw->kern_s[0] = _OF_;
	conv_params_sw->kern_s[1] = _IF_;
	conv_params_sw->kern_s[2] = _FS_;
	conv_params_sw->kern_s[3] = _FS_;
	conv_params_sw->maxog     = _MAX_OG_;

#if _FS_ == 3
	n_weights        = conv_params_hw->pout*conv_params_hw->pin/3*32;
#else
	n_weights        = conv_params_hw->pout*conv_params_hw->pin*32;	
#endif

	SIZE conv_wei_dim = _OF_*_IF_*_FS_*_FS_;
	static DATA conv_wei_array[_OF_*_IF_*_FS_*_FS_];

	SIZE conv_bias_dim = _OF_;
	static DATA conv_bias_array[_OF_];




/*
██╗      ██████╗  █████╗ ██████╗     ███████╗██████╗  ██████╗ ███╗   ███╗    ███████╗██╗██╗     ███████╗
██║     ██╔═══██╗██╔══██╗██╔══██╗    ██╔════╝██╔══██╗██╔═══██╗████╗ ████║    ██╔════╝██║██║     ██╔════╝
██║     ██║   ██║███████║██║  ██║    █████╗  ██████╔╝██║   ██║██╔████╔██║    █████╗  ██║██║     █████╗  
██║     ██║   ██║██╔══██║██║  ██║    ██╔══╝  ██╔══██╗██║   ██║██║╚██╔╝██║    ██╔══╝  ██║██║     ██╔══╝  
███████╗╚██████╔╝██║  ██║██████╔╝    ██║     ██║  ██║╚██████╔╝██║ ╚═╝ ██║    ██║     ██║███████╗███████╗
╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝     ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝    ╚═╝     ╚═╝╚══════╝╚══════╝
                                                                                                        
*/


#ifdef READ_FILES

	sprintf(filename, "%sweights_array_file_hw.bin", load_data_dir);
	load_fixed(filename, n_weights, wPointer);

	sprintf(filename, "%sweights_array_file", load_data_dir);
	load_fixed(filename,conv_wei_dim,conv_wei_array);

	sprintf(filename, "%sbias_array_file", load_data_dir);
	load_fixed(filename,conv_bias_dim,conv_bias_array);

	conv_params_hw->kernel = wPointer;
	conv_params_sw->kernel = conv_wei_array;
	conv_params_sw->bias   = conv_bias_array;

	char *pesi = (char*)malloc(_OF_*_IF_*27*sizeof(char));
	char *bias8 = (char*)malloc(_OF_*sizeof(char));

	if(PRECISION8){
	
		for(o=0; o<_OF_; o++){
	    		for(i=0; i<_IF_/3; i++){
				for(j=0; j<27; j++){
					pesi[j + i*27 + o*_IF_/3*27] = conv_params_sw->kernel[j + i*27 + o*_IF_/3*27];
				}
			}
		}
		conv_params_sw->kernel = (DATA*)pesi;
	}

#else

/*
██╗    ██╗███████╗██╗ ██████╗ ██╗  ██╗████████╗███████╗       ██╗        █████╗  ██████╗████████╗
██║    ██║██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝██╔════╝       ██║       ██╔══██╗██╔════╝╚══██╔══╝
██║ █╗ ██║█████╗  ██║██║  ███╗███████║   ██║   ███████╗    ████████╗    ███████║██║        ██║   
██║███╗██║██╔══╝  ██║██║   ██║██╔══██║   ██║   ╚════██║    ██╔═██╔═╝    ██╔══██║██║        ██║   
╚███╔███╔╝███████╗██║╚██████╔╝██║  ██║   ██║   ███████║    ██████║      ██║  ██║╚██████╗   ██║   
 ╚══╝╚══╝ ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝    ╚═════╝      ╚═╝  ╚═╝ ╚═════╝   ╚═╝   
                                                                                                 
*/

       for (i=0;i<n_weights;i++){
          wPointer[i]=0x0;
        }
               
        for (i=0;i<conv_bias_dim;i++){
          conv_bias_array[i]=0x0;
        }
        
        for (i=0;i<conv_wei_dim;i++){
          conv_wei_array[i]=0x0;
        }

        wPointer[4]       = 0x1000;
        conv_wei_array[4] = 0x1000;


        conv_params_hw->kernel = wPointer;
        conv_params_sw->kernel = conv_wei_array;
	conv_params_sw->bias   = conv_bias_array;


#endif
	
	
	
}
 




	
void cnnMain(DATA* image, float* results)
{
	srand(time(NULL));
        
	SOCMAP soc = socs[0];
	
	SIZE conv_dim = _OF_ * _IH_ * _IW_;
	
	unsigned int padding;
	padding = (_ZERO_PAD_) ? _FS_/ 2 : 0;		
	

	unsigned int iw = _IW_;
      	
	if(PRECISION8){
		while(iw%(4))
          	iw++;
	}
	else{
		while(iw%(2))	//se conv_width == 16 basta che sia multiplo di 2
			iw++;
	}
	
	unsigned int in_s  [3] = {conv_params_hw->pin,_IH_,iw};
	unsigned int out_s [3] = {conv_params_hw->pout,_IH_/_STRIDE_,iw/_STRIDE_};
	SIZE stride[2] = {_STRIDE_,_STRIDE_};
	SIZE pad   [2] = {padding,padding};
	bool activate  = false;

	SIZE in_s_sw  [3] = {_IF_,_IH_,_IW_};
	SIZE out_s_sw [3] = {_OF_,_IH_/_STRIDE_,_IW_/_STRIDE_};
        
	//DATA output[_OF_*_IH_*iw];
	DATA *output = (DATA*)malloc(_OF_*_IH_*iw*sizeof(DATA_CALC));
        
	int conv_id;
	printf("Main!\n");
	
	
	unsigned int i,j;     
        
	DATA_CALC * in;
	in = (DATA_CALC *) soc->in;
	for (j=0;j<_IF_;j++){
		for (i=0;i<_IH_*_IW_;i++){
			if(j<_IF_)
				in[i+j*_IH_*_IW_]=  rand()%32; 
						//rand()%65536; 
						//i*0x1;
			else
			    in[i+j*_IH_*_IW_]= i*0x0;
		  }
		}
	/*
	unsigned int k;
	printf("\nsoc->in\n");
	for (j=0;j<_IF_;j++){
		  for (i=0;i<_IH_;i++){
			for (k=0;k<_IW_;k++){
				printf("%x\t",(unsigned short int)in[i*_IW_ + j*_IH_*_IW_ + k]);
		}
		printf("\n");
	}
	printf("\n");
	}*/


	DATA_CALC* inp = in;

	_tcreate_(time);

	//DATA_CALC input [_IF_*_IH_*iw];
	DATA_CALC *input = (DATA_CALC*)malloc(_IF_*_IH_*iw*sizeof(DATA_CALC));

	//if(iw != _IW_)
	  preprocessing(inp, input, _IW_, iw, _IH_, _IF_);

	//int t_pre = get_wall_time()-time;
	//_tprintf_("\npreprocessing time: %5.3f ms\n", t_pre/1000);
	_tprintf_("\npreprocessing time: %5.3f ms\n", (get_wall_time()-time)/1000);


	for (i=0;i<_IF_*_IH_*iw;i++){
		in[i]= input[i];
	}

           
        for (i=0;i<_OF_*_IH_*iw;i++){
          soc->out[i]=0x0101;
        }

      //  printf("\n size data calc :%d\n",sizeof(DATA_CALC));
	
	//print_data(conv_params_hw->kernel, n_weights, "w_test.txt");
        //print_data(soc->in, _IF_ * _IH_ * _IW_, "x_test.txt");

        _tprintf_("\n\nIF\tOF\tIH\tIW\tmax_OG\tt[ms]\tGOps\tGOpss\n");


        int mog;
        int l;
       // #define DEEP_EXPLORE
#ifdef DEEP_EXPLORE
#define MAXIF 512
#define MAXOF 512
#define MAXSIZE 28

	for (j = _N_COL_*3; j < MAXIF; j += _N_COL_*3){ //if
       
		for (k = _N_ROW_; k < MAXOF; k += _N_ROW_){ //of
			for (l = 8; l < MAXSIZE; l += 2){ //size
				conv_params_hw->pout= k;
				conv_params_hw->pin= j;
				conv_params_hw->kern_s[0] = k;
				conv_params_hw->kern_s[1] = j;

                  
                in_s  [0] = conv_params_hw->pin;
                in_s  [1] = l;
                in_s  [2] = l;
                out_s [0] = conv_params_hw->pout;
                out_s [1] = l;
                out_s [2] = l;
                float min_t=0;
                int maxog_min_t=0;
                for(mog= _MAX_OG_; mog <= max_mog_[l]; mog=mog+_MAX_OG_INC_){


                  conv_params_hw->maxog     = mog;

                  _tcreate_(time_hw);

                  conv_id = spatconv_forward_hw(conv_params_hw, soc->in, soc->out, soc, (SIZE*)in_s, (SIZE*)out_s, stride, pad, activate, _QF_, PRECISION8);
	
	          spatconv_wait(soc, conv_id);
	          
	          float t = get_wall_time()-time_hw;
	          
	          if (mog== _MAX_OG_){
	            min_t=t;
	            maxog_min_t=mog;
	            }
	          else if (t<min_t){
	            min_t=t;
	            maxog_min_t=mog;
	            }

                  if (mog== max_mog_[l]) _tprintf_("%d\t%d\t%d\t%d\t%d\t%5.3f\n", in_s  [0], out_s [0], in_s  [1], in_s  [2], maxog_min_t, (min_t)/1000);

                }
       
       }
    }
    _tprintf_("\n\r");
    }   
       
   #else


/*

██╗      █████╗ ██╗   ██╗███████╗██████╗ ███████╗    ██╗      ██████╗  ██████╗ ██████╗ 
██║     ██╔══██╗╚██╗ ██╔╝██╔════╝██╔══██╗██╔════╝    ██║     ██╔═══██╗██╔═══██╗██╔══██╗
██║     ███████║ ╚████╔╝ █████╗  ██████╔╝███████╗    ██║     ██║   ██║██║   ██║██████╔╝
██║     ██╔══██║  ╚██╔╝  ██╔══╝  ██╔══██╗╚════██║    ██║     ██║   ██║██║   ██║██╔═══╝ 
███████╗██║  ██║   ██║   ███████╗██║  ██║███████║    ███████╗╚██████╔╝╚██████╔╝██║     
╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚══════╝    ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝     
                                                                                       
*/

		for (l=0;l<NLAYERS;l++){
        
			conv_params_hw->pout= of_[l];
			conv_params_hw->pin= if_[l];
			conv_params_hw->kern_s[0] = of_[l];
			conv_params_hw->kern_s[1] = if_[l];


                  
			in_s  [0] = conv_params_hw->pin;
			in_s  [1] = ih_[l];
			in_s  [2] = iw_[l];
			out_s [0] = conv_params_hw->pout;
			out_s [1] = ih_[l];
			out_s [2] = iw_[l];
			float min_t=0;
			int maxog_min_t=0;

			long long int ops  = (long long int)of_[l]*(long long int)if_[l]*(long long int)ih_[l]*(long long int)iw_[l]*(long long int)(_FS_*_FS_*2);
			//printf("%lld\t%ld\t%ld\t%ld\t%ld\t\n", ops,of_[l],if_[l],ih_[l],iw_[l]);
			double gops = (double)ops/1000000000;
			double gopss;

/*

███╗   ███╗ █████╗ ██╗  ██╗ ██████╗  ██████╗     ██╗      ██████╗  ██████╗ ██████╗ 
████╗ ████║██╔══██╗╚██╗██╔╝██╔═══██╗██╔════╝     ██║     ██╔═══██╗██╔═══██╗██╔══██╗
██╔████╔██║███████║ ╚███╔╝ ██║   ██║██║  ███╗    ██║     ██║   ██║██║   ██║██████╔╝
██║╚██╔╝██║██╔══██║ ██╔██╗ ██║   ██║██║   ██║    ██║     ██║   ██║██║   ██║██╔═══╝ 
██║ ╚═╝ ██║██║  ██║██╔╝ ██╗╚██████╔╝╚██████╔╝    ███████╗╚██████╔╝╚██████╔╝██║     
╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝     ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝     
                                                                                   
*/


			for(mog= _MAX_OG_; mog <= max_mog_[l]; mog=mog+_MAX_OG_INC_){


                  conv_params_hw->maxog     = mog;

                  _tcreate_(time_hw);



/*
██╗  ██╗██╗    ██╗     ██████╗ ██████╗ ███╗   ██╗██╗   ██╗
██║  ██║██║    ██║    ██╔════╝██╔═══██╗████╗  ██║██║   ██║
███████║██║ █╗ ██║    ██║     ██║   ██║██╔██╗ ██║██║   ██║
██╔══██║██║███╗██║    ██║     ██║   ██║██║╚██╗██║╚██╗ ██╔╝
██║  ██║╚███╔███╔╝    ╚██████╗╚██████╔╝██║ ╚████║ ╚████╔╝ 
╚═╝  ╚═╝ ╚══╝╚══╝      ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝  ╚═══╝  
                                                          
*/
   
		conv_id = spatconv_forward_hw(conv_params_hw, soc->in, soc->out, soc, (SIZE*)in_s, (SIZE*)out_s, stride, pad, activate, _QF_, PRECISION8);
	
	          spatconv_wait(soc, conv_id);
        
	          float t = get_wall_time()-time_hw;
	          
	          if (mog== _MAX_OG_){
	            min_t=t;
	            maxog_min_t=mog;
	            }
	          else if (t<min_t){
	            min_t=t;
	            maxog_min_t=mog;
	            }
                  gopss = gops*1000*1000/min_t;
                  if (mog== max_mog_[l]) _tprintf_("%d\t%d\t%d\t%d\t%d\t%5.3f\t%5.3f\t%5.3f\n", in_s  [0], out_s [0], in_s  [1], in_s  [2], maxog_min_t, (min_t)/1000,gops, gopss);

                } // maxog loop
        }// layers loop
#endif




/*
███████╗██╗    ██╗     ██████╗ ██████╗ ███╗   ██╗██╗   ██╗
██╔════╝██║    ██║    ██╔════╝██╔═══██╗████╗  ██║██║   ██║
███████╗██║ █╗ ██║    ██║     ██║   ██║██╔██╗ ██║██║   ██║
╚════██║██║███╗██║    ██║     ██║   ██║██║╚██╗██║╚██╗ ██╔╝
███████║╚███╔███╔╝    ╚██████╗╚██████╔╝██║ ╚████║ ╚████╔╝ 
╚══════╝ ╚══╝╚══╝      ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝  ╚═══╝  
                                                          
*/

	_tprintf_("\n\nSoftware conv...\n\n");
	spatconv_forward_sw(conv_params_sw, soc->in, output, in_s_sw, out_s_sw, stride, pad, activate,_N_COL_, _QF_,0);
        
	
	unsigned int _OW_=   iw-(_FS_-1)*(1-_ZERO_PAD_);
   	unsigned int _OH_= _IH_-(_FS_-1)*(1-_ZERO_PAD_);

	/*printf("\n output \n");
	for(o=0; o<_OF_;o++){
        for(i=0; i<_OH_; i++){
          for(j=0; j<_OW_; j++){
            printf("%x\t",(unsigned short int)output[o*_OH_*_OW_ + i*_OW_ + j]);
          }
          printf("\n");
        }
        printf("\n");
      }*/

	DATA_CALC* out_sw = (DATA_CALC*) output;	
	DATA_CALC* out_hw = (DATA_CALC*) soc->out;

	/*printf("\n out sw \n");
	for(o=0; o<_OF_;o++){
        for(i=0; i<_OH_; i++){
          for(j=0; j<_OW_; j++){
            printf("%x\t",(unsigned short int)out_sw[o*_OH_*_OW_ + i*_OW_ + j]);
          }
          printf("\n");
        }
        printf("\n");
      }*/

	/*printf("\n out hw \n");
	for(o=0; o<_OF_;o++){
        for(i=0; i<_OH_; i++){
          for(j=0; j<_OW_; j++){
            printf("%x\t",(unsigned short int)out_hw[o*_OH_*_OW_ + i*_OW_ + j]);
          }
          printf("\n");
        }
        printf("\n");
      }*/


	//DATA_CALC out_sw_post [_OF_*_OH_*_OW_];
	DATA_CALC* out_sw_post = (DATA_CALC*)malloc(_OF_*_OH_*_OW_*sizeof(DATA_CALC));
	//if(iw != _IW_)
	  postprocessing(out_sw, out_sw_post, _IW_, _OW_, _OH_, _OF_);
	
	_tcreate_(time2);

	//DATA_CALC out_hw_post [_OF_*_OH_*_OW_];
	DATA_CALC* out_hw_post = (DATA_CALC*)malloc(_OF_*_OH_*_OW_*sizeof(DATA_CALC));
	//if(iw != _IW_)
	  postprocessing(out_hw, out_hw_post, _IW_, _OW_, _OH_, _OF_);

	_tprintf_("\npostprocessing time: %5.3f ms\n", (get_wall_time()-time2)/1000);


/*
 ██████╗██╗  ██╗███████╗ ██████╗██╗  ██╗
██╔════╝██║  ██║██╔════╝██╔════╝██║ ██╔╝
██║     ███████║█████╗  ██║     █████╔╝ 
██║     ██╔══██║██╔══╝  ██║     ██╔═██╗ 
╚██████╗██║  ██║███████╗╚██████╗██║  ██╗
 ╚═════╝╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝
                                        
*/


	int checksum_hw=0, checksum_sw=0;
	int e=0;
	#define MAXERR 10
	#define MAXCOR 10
        
	DATA_CALC * sw_out;
	DATA_CALC * hw_out;
        
	sw_out = (DATA_CALC *) /*output*/ out_sw_post;
	hw_out = (DATA_CALC *) /*soc->out*/ out_hw_post;

		

        for (i=0; i<conv_dim/(stride[0]*stride[1]);i++){
           if (hw_out[i]!=sw_out[i]){
             if (e< MAXERR )//|| hw_out[i] - sw_out[i]>10 || sw_out[i]-hw_out[i] >10)

               printf("%05d HW: 0x%08x != SW: 0x%08x\n", i, hw_out[i] ,sw_out[i]);
               
             e++;
           }
           else if (i<MAXCOR)
             printf("          %05d HW: 0x%08x == SW: 0x%08x\n", i, hw_out[i] ,sw_out[i]);
          checksum_hw+=hw_out[i];
          checksum_sw+=sw_out[i];
        }

		
	
        printf("\ntotal errors: %05d\n", e );
        printf("checksum_sw: %d\n", checksum_sw );
        printf("checksum_hw: %d\n", checksum_hw );
        printf("avg_err: %f\n", (float)(checksum_sw-checksum_hw)/(float)e );
        
        print_data((short int*)output, conv_dim/(stride[0]*stride[1]), "socout.txt");
        print_data(soc->out, conv_dim/(stride[0]*stride[1]), "output.txt");
        
        printf("Convdim: %d\n", conv_dim);
        
//	free(output);
//	free(input);
	free(out_sw_post);
	free(out_hw_post);


	remove("/tmp/lockPL.neuraghe");
        exit(0);
	          
	
                  
}





void preprocessing(DATA_CALC* pre, DATA_CALC* pre_out, unsigned int IW, unsigned int iw, unsigned int IH, unsigned int IF){
  
  unsigned int in,i,j,k;
  
  for(in=0; in<IF;in++){
        for(i=0; i<IH; i++){
          for(j=0; j<IW; j++){
            pre_out[in*IH*iw + i*iw + j] = pre[in*IH*IW + i*IW + j];
          }
          for(k=0; k<iw-IW; k++)
            pre_out[in*IH*iw + i*iw + IW + k] = 0;
        }
      }
}

void postprocessing(DATA_CALC* post, DATA_CALC* post_out, unsigned int IW, unsigned int OW, unsigned int OH, unsigned int OF){
  
  unsigned int o,i,j;
  
  for(o=0; o<OF;o++){
        for(i=0; i<OH; i++){
          for(j=0; j<(OW-(OW-IW)); j++){
            post_out[o*OH*(OW-(OW-IW)) + i*(OW-(OW-IW)) + j] = post[o*OH*OW + i*OW + j];
          }
        }
      }
}

void init_platform(char* bitstream){
	init_soc(socs, &wPointer, _MAXMEM_, 0, bitstream);
}
void free_platform(){
	munmap_soc(socs);
	int i;	
	for (i=0; i<_NUM_CLUSTER_; i++)
		free(socs[i]);
}

