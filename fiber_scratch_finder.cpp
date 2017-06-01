#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv; 

#include "img_region_grow.h"
#include "fast_thin.h"
#include "img_io_trans.h"

#define NOISE_CULL_TH 30
#define  THIN_ITER     5 
#define  HOUGH_TH      40
#define  LINES_NUM_ALLOW 15
#define  SCRATCH_OFFSET_SIDE   1
//#define  REPORT
//#define  LINE_DEBUG
struct sort_elm 
{
	int elm;
	int elm_local;
};
sort_elm tmp_sto_sort;

////////////////////////////////////////////////////////////////////////////////
//main implementaion


inline bool elmSmall(const float& lhs, const float& rhs)
{
	return lhs < rhs;
}

int ImageAdjust(Mat src,
				double low=0.6645, double high=0.9332,   // X����low and high are the intensities of src
				double bottom=0, double top=1, // Y����mapped to bottom and top of dst
				double gamma=1 )
{
	if(   low<0 && low>1 && high <0 && high>1&&
		bottom<0 && bottom>1 && top<0 && top>1 && low>high)
		return -1;

	for(int i=0;i<src.rows;i++)
		for(int j=0;j<src.cols;j++)
		{
			float min_in=(float)high;
			if(src.at<float>(i,j)<min_in)
				min_in=src.at<float>(i,j);
			float max_in=(float)low;
			if(min_in>max_in)
				max_in=min_in;
			src.at<float>(i,j)=max_in;
			float dst_out=(float)((max_in-low)/(high-low));
			src.at<float>(i,j)=dst_out;
		}


		return 0;
}


int cvnormalize8(Mat src,Mat* dst,int flag)
{
	double min_val=-1;
	double max_val=-1;
	minMaxLoc(src,&min_val,&max_val);




	//floatתfloat
	if(flag==0){
		for(int i=0;i<src.rows;i++)
			for(int j=0;j<src.cols;j++)
			{
				double stosrc=(src.at<float>(i,j)-min_val)/(max_val-min_val);
				src.at<float>(i,j)=(float)stosrc;
			}
	}

	//ucharתuchar������ԭͼƬ
	else if(flag==1){
		for(int i=0;i<src.rows;i++)
			for(int j=0;j<src.cols;j++)
			{
				double stosrc=(src.at<uchar>(i,j)-min_val)/(max_val-min_val)*255.0;
				src.at<uchar>(i,j)=(uchar)(stosrc+0.5);
			}
	}

	//floatתuchar
	if(flag==2){
		for(int i=0;i<src.rows;i++)
			for(int j=0;j<src.cols;j++)
			{
				double stosrc=(src.at<float>(i,j)-min_val)/(max_val-min_val)*255.0;
				(*dst).at<uchar>(i,j)=int(stosrc+0.5);
			}
	}


	//intתfloat
	if(flag==3){
		for(int i=0;i<src.rows;i++)
			for(int j=0;j<src.cols;j++)
			{
				double stosrc=(src.at<int>(i,j)-min_val)/(max_val-min_val);
				(*dst).at<float>(i,j)=(float)stosrc;
			}
	}




	return 0;
}

inline bool ranksort_elmSmall(const sort_elm& lhs, const sort_elm& rhs)
{
	return lhs.elm < rhs.elm;
}
int rank_normalize(Mat src,Mat* dst)
{

	vector<sort_elm> rank_sort;
	for(int i=0;i<src.rows;i++)
		for(int j=0;j<src.cols;j++)
		{

			tmp_sto_sort.elm=(int)src.at<uchar>(i,j);
			tmp_sto_sort.elm_local=j+i*src.cols;
			rank_sort.push_back(tmp_sto_sort);
		}
		std::sort(rank_sort.begin(),rank_sort.end(),ranksort_elmSmall);

		for(int i=0;i<src.rows*src.cols;i++)
			rank_sort[rank_sort[i].elm_local].elm=i;


		Mat dst_pre=Mat::Mat(src.rows,src.cols,CV_32SC1);

		for(int i=0;i<src.rows;i++)
			for(int j=0;j<src.cols;j++)
			{
				dst_pre.at<int>(i,j)=rank_sort[i*src.cols+j].elm;
			}
			cvnormalize8(dst_pre,dst,3);

			return 0;
}


int histtruncate(Mat src,Mat &dst, double lHistCut=0.2,double uHistCut=0.2)
{

	//����һ�γ������ò�ֵ�İ취����gv1��gv2������ûʵ�֣�����matlab�Ĳ�������
	/*
	int m=src.rows*src.cols;
	//double x[329*329+2];

	for(int i=1;i<329*329+2-1;i++)
	{
	x[i]=100*(0.5*i-0.5)/m;
	}
	x[0]=0;
	x[329*329+2-1]=100;

	vector<float> src_one;
	for(int i=0;i<src.rows;i++)
	for(int j=0;j<src.cols;j++)
	src_one.push_back(src.at<float>(i,j));

	std::sort(src_one.begin(),src_one.end(),elmSmall);
	src_one.push_back(src_one[src_one.size()-1]);
	src_one.insert(src_one.begin(),src_one[0]);

	for(unsigned int i=0;i<src_one.size();i++)
	sortv[i]=src_one[i];
	*/

	cvnormalize8(src,&Mat(),0);
	ImageAdjust(src);
	Mat proc_out=Mat::Mat(src.rows,src.cols,CV_8UC1);
	cvnormalize8(src,&proc_out,2);
	Mat proc_out_rank=Mat::Mat(src.rows,src.cols,CV_32FC1);
	rank_normalize(proc_out,&proc_out_rank);
	cvnormalize8(proc_out_rank,&dst,2);
	return 0;
}



int nonlocalmeans_Proc(Mat& src)
{
	cvnormalize8(src,&Mat(),1);

	Mat fB=Mat::Mat(src.rows,src.cols,CV_32FC1);
	Mat fC=Mat::Mat(src.rows,src.cols,CV_32FC1);

	Mat fA;
	src.convertTo(fA,CV_32FC1);
	fA=fA+1;
	log(fA,fB);

	Mat A=src;
	Mat A_gauss;

	GaussianBlur(A,A_gauss,Size(21,21),10.5,10.5);

	A_gauss.convertTo(fA,CV_32FC1);
	fA=fA+1;
	log(fA,fC);
	subtract(fB,fC,fA);

	histtruncate(fA,src);

	return 0;
}

int cull_tinyregion(vector<vector<Point>>& fiber_noise_unidom,Mat& fiber_img,int cullsize)
{

	for(unsigned int i=0;i<fiber_noise_unidom.size();i++)
		if(fiber_noise_unidom[i].size()<(unsigned)cullsize)
		{
		 for(unsigned int j=0;j<fiber_noise_unidom[i].size();j++)
			 fiber_img.at<uchar>(fiber_noise_unidom[i][j].y,fiber_noise_unidom[i][j].x)=255;
		}

	return 0;
}

int BresenhamLine(vector<Point>* line_pt,int x1,int y1,int x2,int y2)
{
	int validpix_num=0;

	int con=1;//�����б�����һ���͵������con=1Ϊб����1-0��
	if(x1>x2)//ʹ������Ϊ��������
	{
		int r=x1;x1=x2;x2=r;
		r=y1;y1=y2;y2=r;
	}
	if(y2<y1)
	{
		y1=-y1,y2=-y2; //ʹ�����x���ԳƱ任
		if(fabs(double(y2-y1))>fabs(double(x1-x2)))//б��ΪС��-1 
		{
			con=4;
			int r=x1;x1=y1;y1=r;
			r=x2;x2=y2;y2=r;
		}
		else //б����-1��0��
			con=3;
	}
	else if(y2-y1>x2-x1)//б�ʴ���1 ʹ֮����y=x�ԳƱ任
	{
		con=2;
		int r=x1;x1=y1,y1=r;
		r=x2;x2=y2,y2=r;   
	}
	int x=x1,y=y1,dx,dy,p;
	dx=x2-x1;
	dy=y2-y1;
	p=2*dy-dx;
	for(;x<=x2;x++)
	{
		switch(con)//��ԭΪ�ʵ�����
		{
		case 2://SetPixel(dc,y,x,color);
			//circle(draw_line_bh,Point(y,x),1,Scalar(0));
			line_pt->push_back(Point(y,x));
			break;

		case 3://SetPixel(dc,x,-y,color);
			//circle(draw_line_bh,Point(x,-y),1,Scalar(0));
			line_pt->push_back(Point(x,-y));
			break;
		case 4://SetPixel(dc,y,-x,color);
			//circle(draw_line_bh,Point(y,-x),1,Scalar(0));
			line_pt->push_back(Point(y,-x));
			break;
		default://SetPixel(dc,x,y,color);
			//circle(draw_line_bh,Point(x,y),1,Scalar(0));
			line_pt->push_back(Point(x,y));
		}
		if(p>=0)
		{
			y++;
			p+=2*(dy-dx);
		}
		else
			p+=2*dy;
	}

	return validpix_num;
}

int line_pt_generate(vector<Vec4i>& lines,int line_size_refine,
					 vector<vector<Point>>& line_side,
					 vector<vector<Point>>& line_side_i_1,
					 vector<vector<Point>>& line_side_i_2,
					 vector<vector<Point>>& line_side_i_3,
					 vector<vector<Point>>& line_side_i_4,
					 vector<vector<Point>>& line_side_i_5,
					 vector<vector<Point>>& line_side_j_1,
					 vector<vector<Point>>& line_side_j_2,
					 vector<vector<Point>>& line_side_j_3,
					 vector<vector<Point>>& line_side_j_4,
					 vector<vector<Point>>& line_side_j_5)

{


	//BresenhamLine�������Ը���ͼ����ֱ�ߵ������˵�õ���ֱ�������еĵ�
	//��bh�����������߶Σ����ҵ��߶��ϵĵ�
	for(int i=0;i<line_size_refine;i++)
	{
		vector<Point>* line_pt=&line_side[i];
		BresenhamLine(line_pt,lines[i][0],lines[i][1],lines[i][2],lines[i][3]);

		vector<Point>* line_pt_side_i_1=&line_side_i_1[i];
		vector<Point>* line_pt_side_i_2=&line_side_i_2[i];
		vector<Point>* line_pt_side_i_3=&line_side_i_3[i];
		vector<Point>* line_pt_side_i_4=&line_side_i_4[i];
		vector<Point>* line_pt_side_i_5=&line_side_i_5[i];

		vector<Point>* line_pt_side_j_1=&line_side_j_1[i];
		vector<Point>* line_pt_side_j_2=&line_side_j_2[i];
		vector<Point>* line_pt_side_j_3=&line_side_j_3[i];
		vector<Point>* line_pt_side_j_4=&line_side_j_4[i];
		vector<Point>* line_pt_side_j_5=&line_side_j_5[i];


		//�ж��߶�����x������y����x1-x2>y1-y2,x����y����
		if(abs(lines[i][0]-lines[i][2])>abs(lines[i][1]-lines[i][3]))//x>y��ֱ������ˮƽ����y���ƶ��õ������ֱ��
		{
		BresenhamLine(line_pt_side_i_1,lines[i][0],lines[i][1]+SCRATCH_OFFSET_SIDE-1,lines[i][2],lines[i][3]+SCRATCH_OFFSET_SIDE-1);
		BresenhamLine(line_pt_side_j_1,lines[i][0],lines[i][1]-SCRATCH_OFFSET_SIDE+1,lines[i][2],lines[i][3]-SCRATCH_OFFSET_SIDE+1);

		BresenhamLine(line_pt_side_i_2,lines[i][0],lines[i][1]+SCRATCH_OFFSET_SIDE-2,lines[i][2],lines[i][3]+SCRATCH_OFFSET_SIDE-2);
		BresenhamLine(line_pt_side_j_2,lines[i][0],lines[i][1]-SCRATCH_OFFSET_SIDE+2,lines[i][2],lines[i][3]-SCRATCH_OFFSET_SIDE+2);

		BresenhamLine(line_pt_side_i_3,lines[i][0],lines[i][1]+SCRATCH_OFFSET_SIDE-3,lines[i][2],lines[i][3]+SCRATCH_OFFSET_SIDE-3);
		BresenhamLine(line_pt_side_j_3,lines[i][0],lines[i][1]-SCRATCH_OFFSET_SIDE+3,lines[i][2],lines[i][3]-SCRATCH_OFFSET_SIDE+3);

		BresenhamLine(line_pt_side_i_4,lines[i][0],lines[i][1]+SCRATCH_OFFSET_SIDE-4,lines[i][2],lines[i][3]+SCRATCH_OFFSET_SIDE-4);
		BresenhamLine(line_pt_side_j_4,lines[i][0],lines[i][1]-SCRATCH_OFFSET_SIDE+4,lines[i][2],lines[i][3]-SCRATCH_OFFSET_SIDE+4);

		BresenhamLine(line_pt_side_i_5,lines[i][0],lines[i][1]+SCRATCH_OFFSET_SIDE-4,lines[i][2],lines[i][3]+SCRATCH_OFFSET_SIDE-4);
		BresenhamLine(line_pt_side_j_5,lines[i][0],lines[i][1]-SCRATCH_OFFSET_SIDE+4,lines[i][2],lines[i][3]-SCRATCH_OFFSET_SIDE+4);

		}
		else 
		{
		BresenhamLine(line_pt_side_i_1,lines[i][0]+SCRATCH_OFFSET_SIDE-1,lines[i][1],lines[i][2]+SCRATCH_OFFSET_SIDE-1,lines[i][3]);
		BresenhamLine(line_pt_side_j_1,lines[i][0]-SCRATCH_OFFSET_SIDE+1,lines[i][1],lines[i][2]-SCRATCH_OFFSET_SIDE+1,lines[i][3]);

		BresenhamLine(line_pt_side_i_2,lines[i][0]+SCRATCH_OFFSET_SIDE-2,lines[i][1],lines[i][2]+SCRATCH_OFFSET_SIDE-2,lines[i][3]);
		BresenhamLine(line_pt_side_j_2,lines[i][0]-SCRATCH_OFFSET_SIDE+2,lines[i][1],lines[i][2]-SCRATCH_OFFSET_SIDE+2,lines[i][3]);

		BresenhamLine(line_pt_side_i_3,lines[i][0]+SCRATCH_OFFSET_SIDE-3,lines[i][1],lines[i][2]+SCRATCH_OFFSET_SIDE-3,lines[i][3]);
		BresenhamLine(line_pt_side_j_3,lines[i][0]-SCRATCH_OFFSET_SIDE+3,lines[i][1],lines[i][2]-SCRATCH_OFFSET_SIDE+3,lines[i][3]);

		BresenhamLine(line_pt_side_i_4,lines[i][0]+SCRATCH_OFFSET_SIDE-4,lines[i][1],lines[i][2]+SCRATCH_OFFSET_SIDE-4,lines[i][3]);
		BresenhamLine(line_pt_side_j_4,lines[i][0]-SCRATCH_OFFSET_SIDE+4,lines[i][1],lines[i][2]-SCRATCH_OFFSET_SIDE+4,lines[i][3]);

		BresenhamLine(line_pt_side_i_5,lines[i][0]+SCRATCH_OFFSET_SIDE-5,lines[i][1],lines[i][2]+SCRATCH_OFFSET_SIDE-4,lines[i][3]);
		BresenhamLine(line_pt_side_j_5,lines[i][0]-SCRATCH_OFFSET_SIDE+5,lines[i][1],lines[i][2]-SCRATCH_OFFSET_SIDE+4,lines[i][3]);

		}

	}

	return 0;
}

int decide_grad(float gi5,
			float gi4,
			float gi3,
			float gi2,
			float gi1,
			float gij,
			float gj1,
			float gj2,
			float gj3,
			float gj4,
			float gj5,
			int* flag
			)
{
	float g[11]={0};
	g[0]=gi5;g[1]=gi4;g[2]=gi3;g[3]=gi2;g[4]=gi1;g[5]=gij;
	g[6]=gj1;g[7]=gj2;g[8]=gj3;g[9]=gj4;g[10]=gj5;


	/*-------���ݶ���ֵ--------*/
	float max_line_grad=-1;
	float min_line_grad=99999;
	for(int i=0;i<11;i++)
	{
	if(g[i]>=max_line_grad)
		max_line_grad=g[i];
	if(g[i]<min_line_grad)
		min_line_grad=g[i];
	}


	/*------�ҵ������м䲿�ݵ��ݶ���ֵ��һ�㴦���м��߶ε����࣬���ҵ�line4,line5,line6�������ߵ���һ�����ݶ�ֵ���--------*/
	float middle_max_grad=-1;
	int middle_max_grad_id=0;
	for(int i=0;i<3;i++)
		if(g[4+i]>=middle_max_grad)
		{
		   middle_max_grad_id=4+i;
		   middle_max_grad=g[4+i];
		}
	

	/*--------�ж����⺬ԭʼֱ�ߵ�10��ֱ�ϣ��ݶȵĲ����Ƿ����˫��ķֲ���---------*/
	//�����ֵ�ͱ�׼��
	double mean_line_grad=0;
	double diff_line_grad=0;
	for(int i=0;i<11;i++)
		mean_line_grad+=g[i];
		mean_line_grad/=11;

	for(int i=0;i<11;i++)
		diff_line_grad+=(g[i]-mean_line_grad)*(g[i]-mean_line_grad);
	diff_line_grad/=11;
	diff_line_grad=sqrt(diff_line_grad);

	int sign_change=0;
	int cur_sign=0;
	int pre_sign=0;

	if(g[1]-g[0]>=0)
		pre_sign=1;
	else pre_sign=-1;

	//ͳ���ݶȷֲ��Ĳ��塢���ȵı仯���
	for(int i=0;i<(11-1);i++)
	{
		if(g[i+1]-g[i]>=0)
		cur_sign=1;
		else cur_sign=-1;

		if(cur_sign!=pre_sign)//��������һ�����Ȼ򲨷�
			sign_change++;

		pre_sign=cur_sign;
	}


	//����ݶȷֲ��Ĳ��塢���ȵı仯����3�Σ�˵���ݶȷֲ�����˫��ֲ�����Ϊֱ���������Ļ��ۣ�������Ϊֱ�����ڻ��ۺۼ�����˫����ݶȱ仯����˫��ֲ���
	//�ݶ������Сֵ����5��˵���ݶȱ仯�ϴ����ݶȷֲ�����˫��ֲ���������ж�Ϊ�����Ļ���
	  if(sign_change==3)
		if(abs(max_line_grad-min_line_grad)>5)
		{
			*flag=1;
			//printf("��⵽˫��,�ݶ��ź���Ϊ��");	
		}

		//�ݶ������Сֵ����8��˵���ݶȱ仯�ܴ��������Ļ���
		if(abs(max_line_grad-min_line_grad)>8)
		{
			*flag=1;
		//	printf("�ݶ��ź���Ϊ��\n");
		}


	return 0;
}


int decide_gray(float line_mean_gray,float line_mean_gray_i_1,
				float line_mean_gray_j_1,float line_mean_gray_i_5,float line_mean_gray_j_5,
				float both_side_gray_diff_pro_i,float both_side_gray_diff_pro_j,int* gray_flag)

{
	float max_gray=-1;
	if(line_mean_gray>max_gray)max_gray=line_mean_gray;
	if(line_mean_gray_i_1>max_gray)max_gray=line_mean_gray_i_1;
	if(line_mean_gray_j_1>max_gray)max_gray=line_mean_gray_j_1;
	
	
	if(both_side_gray_diff_pro_i>0.7 || both_side_gray_diff_pro_j>0.7 ||
		(both_side_gray_diff_pro_i>0.6 && both_side_gray_diff_pro_j>0.6))
	{

		if(max_gray-line_mean_gray_i_5>1 && max_gray-line_mean_gray_j_5>1)//�;�����Զ���߻ҶȾ�ֵ���бȽ�
			if(max_gray*2-line_mean_gray_i_5-line_mean_gray_j_5>2)
			{
				*gray_flag=1;
				//printf("ͨ���Ҷ��źż����1\n");
			}	
	}

	if(both_side_gray_diff_pro_i>0.5 && both_side_gray_diff_pro_j>0.5 ||
		(both_side_gray_diff_pro_i>0.8 || both_side_gray_diff_pro_j>0.8))
		if(max_gray-line_mean_gray_i_1>1 && max_gray-line_mean_gray_i_1>1)//�;���������߻ҶȾ�ֵ���бȽ�
			if(max_gray*2-line_mean_gray_i_1-line_mean_gray_j_1>2)
			{
				*gray_flag=1;
				//printf("ͨ���Ҷ��źż����2\n");
			}	

		
	return 0;
}
int cull_lines(Mat& fiber_hat,
					 vector<vector<Point>>& line_side,
					 vector<vector<Point>>& line_side_i_1,
					 vector<vector<Point>>& line_side_i_2,
					 vector<vector<Point>>& line_side_i_3,
					 vector<vector<Point>>& line_side_i_4,
					 vector<vector<Point>>& line_side_i_5,
					 vector<vector<Point>>& line_side_j_1,
					 vector<vector<Point>>& line_side_j_2,
					 vector<vector<Point>>& line_side_j_3,
					 vector<vector<Point>>& line_side_j_4,
					 vector<vector<Point>>& line_side_j_5,
					 Mat& fiber_grad_map,vector<int>* corret_id)
{
	vector<int> sto_linealong_gray;
	vector<int> sto_linealong_gray_i_1;
	vector<int> sto_linealong_gray_i_2;
	vector<int> sto_linealong_gray_i_3;
	vector<int> sto_linealong_gray_i_4;
	vector<int> sto_linealong_gray_i_5;

	vector<int> sto_linealong_gray_j_1;
	vector<int> sto_linealong_gray_j_2;
	vector<int> sto_linealong_gray_j_3;
	vector<int> sto_linealong_gray_j_4;
	vector<int> sto_linealong_gray_j_5;

	float sto_line_grad;
	float sto_line_grad_i_1;
	float sto_line_grad_i_2;
	float sto_line_grad_i_3;
	float sto_line_grad_i_4;
	float sto_line_grad_i_5;

	float sto_line_grad_j_1;
	float sto_line_grad_j_2;
	float sto_line_grad_j_3;
	float sto_line_grad_j_4;
	float sto_line_grad_j_5;

	//����ÿһ��ֱ��
	for(unsigned int i=0;i<line_side.size();i++)
	{

		sto_linealong_gray.clear();
		sto_linealong_gray_i_1.clear();
		sto_linealong_gray_i_2.clear();
		sto_linealong_gray_i_3.clear();
		sto_linealong_gray_i_4.clear();
		sto_linealong_gray_i_5.clear();

		sto_linealong_gray_j_1.clear();
		sto_linealong_gray_j_2.clear();
		sto_linealong_gray_j_3.clear();
		sto_linealong_gray_j_4.clear();
		sto_linealong_gray_j_5.clear();

		sto_line_grad=0;
		sto_line_grad_i_1=0;
		sto_line_grad_i_2=0;
		sto_line_grad_i_3=0;
		sto_line_grad_i_4=0;
		sto_line_grad_i_5=0;

		sto_line_grad_j_1=0;
		sto_line_grad_j_2=0;
		sto_line_grad_j_3=0;
		sto_line_grad_j_4=0;
		sto_line_grad_j_5=0;
		//����ÿһ��ֱ���ϵ�ÿһ����
		for(unsigned int j=0;j<line_side[i].size();j++)
		{

			//��fiber_grad_map��ȡ��ÿ��ֱ�ߵ��ݶ�ֵ
			sto_line_grad+=fiber_grad_map.at<float>(line_side[i][j].y,line_side[i][j].x);
			sto_line_grad_i_1+=fiber_grad_map.at<float>(line_side_i_1[i][j].y,line_side_i_1[i][j].x);
			sto_line_grad_i_2+=fiber_grad_map.at<float>(line_side_i_2[i][j].y,line_side_i_2[i][j].x);
			sto_line_grad_i_3+=fiber_grad_map.at<float>(line_side_i_3[i][j].y,line_side_i_3[i][j].x);
			sto_line_grad_i_4+=fiber_grad_map.at<float>(line_side_i_4[i][j].y,line_side_i_4[i][j].x);
			sto_line_grad_i_5+=fiber_grad_map.at<float>(line_side_i_5[i][j].y,line_side_i_5[i][j].x);
			sto_line_grad_j_1+=fiber_grad_map.at<float>(line_side_j_1[i][j].y,line_side_j_1[i][j].x);
			sto_line_grad_j_2+=fiber_grad_map.at<float>(line_side_j_2[i][j].y,line_side_j_2[i][j].x);
			sto_line_grad_j_3+=fiber_grad_map.at<float>(line_side_j_3[i][j].y,line_side_j_3[i][j].x);
			sto_line_grad_j_4+=fiber_grad_map.at<float>(line_side_j_4[i][j].y,line_side_j_4[i][j].x);
			sto_line_grad_j_5+=fiber_grad_map.at<float>(line_side_j_5[i][j].y,line_side_j_5[i][j].x);

			//��fiber_hat��ȡ��ÿ��ֱ�ߵĻҶ�ֵ
			sto_linealong_gray.push_back(fiber_hat.at<uchar>(line_side[i][j].y,line_side[i][j].x));
			sto_linealong_gray_i_1.push_back(fiber_hat.at<uchar>(line_side_i_1[i][j].y,line_side_i_1[i][j].x));
			sto_linealong_gray_i_2.push_back(fiber_hat.at<uchar>(line_side_i_2[i][j].y,line_side_i_2[i][j].x));
			sto_linealong_gray_i_3.push_back(fiber_hat.at<uchar>(line_side_i_3[i][j].y,line_side_i_3[i][j].x));
			sto_linealong_gray_i_4.push_back(fiber_hat.at<uchar>(line_side_i_4[i][j].y,line_side_i_4[i][j].x));
			sto_linealong_gray_i_5.push_back(fiber_hat.at<uchar>(line_side_i_5[i][j].y,line_side_i_5[i][j].x));
			sto_linealong_gray_j_1.push_back(fiber_hat.at<uchar>(line_side_j_1[i][j].y,line_side_j_1[i][j].x));
			sto_linealong_gray_j_2.push_back(fiber_hat.at<uchar>(line_side_j_2[i][j].y,line_side_j_2[i][j].x));
			sto_linealong_gray_j_3.push_back(fiber_hat.at<uchar>(line_side_j_3[i][j].y,line_side_j_3[i][j].x));
			sto_linealong_gray_j_4.push_back(fiber_hat.at<uchar>(line_side_j_4[i][j].y,line_side_j_4[i][j].x));
			sto_linealong_gray_j_5.push_back(fiber_hat.at<uchar>(line_side_j_5[i][j].y,line_side_j_5[i][j].x));


		}


#ifdef REPORT
		cout<<"\n";
		cout<<"			current detect line id is"<<i<<"\n";
		cout<<"/*--------------�ݶ�ֵ�������------------*/"<<"\n";
		cout<<"����5�߶ε��ݶ���ֵ��"<<sto_line_grad_i_5/(double)line_side_i_5[i].size()<<"\n";
		cout<<"����4�߶ε��ݶ���ֵ��"<<sto_line_grad_i_4/(double)line_side_i_4[i].size()<<"\n";
		cout<<"����3�߶ε��ݶ���ֵ��"<<sto_line_grad_i_3/(double)line_side_i_3[i].size()<<"\n";
		cout<<"����2�߶ε��ݶ���ֵ��"<<sto_line_grad_i_2/(double)line_side_i_2[i].size()<<"\n";
		cout<<"����1�߶ε��ݶ���ֵ��"<<sto_line_grad_i_1/(double)line_side_i_1[i].size()<<"\n";
		cout<<"�߶ε��ݶ���ֵ��"<<sto_line_grad/(double)line_side[i].size()<<"\n";
		cout<<"����1�߶ε��ݶ���ֵ��"<<sto_line_grad_j_1/(double)line_side_j_1[i].size()<<"\n";
		cout<<"����2�߶ε��ݶ���ֵ��"<<sto_line_grad_j_2/(double)line_side_j_2[i].size()<<"\n";
		cout<<"����3�߶ε��ݶ���ֵ��"<<sto_line_grad_j_3/(double)line_side_j_3[i].size()<<"\n";
		cout<<"����4�߶ε��ݶ���ֵ��"<<sto_line_grad_j_4/(double)line_side_j_4[i].size()<<"\n";
		cout<<"����5�߶ε��ݶ���ֵ��"<<sto_line_grad_j_5/(double)line_side_j_5[i].size()<<"\n";
#endif

		/*--------ͨ��ԭʼֱ��������ݶ�ֵ���ж�ֱ�ߵ�����-----*/
		int grad_flag=0;
		decide_grad(sto_line_grad_i_5/(float)line_side_i_5[i].size(),
			sto_line_grad_i_4/(float)line_side_i_4[i].size(),
			sto_line_grad_i_3/(float)line_side_i_3[i].size(),
			sto_line_grad_i_2/(float)line_side_i_2[i].size(),
			sto_line_grad_i_1/(float)line_side_i_1[i].size(),
			sto_line_grad/(float)line_side[i].size(),
			sto_line_grad_j_1/(float)line_side_j_1[i].size(),
			sto_line_grad_j_2/(float)line_side_j_2[i].size(),
			sto_line_grad_j_3/(float)line_side_j_3[i].size(),
			sto_line_grad_j_4/(float)line_side_j_4[i].size(),
			sto_line_grad_j_5/(float)line_side_j_5[i].size(),
			&grad_flag
			);

		/*--------ͨ��ԭʼֱ������ĻҶ�ֵ���ж�ֱ�ߵ�����-----*/
		//�����߶��ϵ�ƽ���Ҷ�ֵ����˫�������Զ��ֱ�ߵĻҶȾ�ֵ
		float line_mean_gray=0;
		float line_mean_gray_i_1=0;
		float line_mean_gray_j_1=0;
		float line_mean_gray_i_5=0;
		float line_mean_gray_j_5=0;
		
		for (unsigned int k=0;k<sto_linealong_gray.size();k++)
		{
		line_mean_gray+=sto_linealong_gray[k];
		line_mean_gray_i_1+=sto_linealong_gray_i_1[k];
		line_mean_gray_j_1+=sto_linealong_gray_j_1[k];
		line_mean_gray_i_5+=sto_linealong_gray_i_5[k];
		line_mean_gray_j_5+=sto_linealong_gray_j_5[k];
		}
		line_mean_gray/=sto_linealong_gray.size();
		line_mean_gray_i_1/=sto_linealong_gray_i_1.size();
		line_mean_gray_j_1/=sto_linealong_gray_j_1.size();
		line_mean_gray_i_5/=sto_linealong_gray_i_5.size();
		line_mean_gray_j_5/=sto_linealong_gray_j_5.size();

		//�Ա�ԭʼ�߶���ÿһ��ĻҶ�ֵ��˫��(������Զ��)�߶��ϻҶ�ֵ��������Ϊ��������»�����ÿһ��ĻҶ�ֵ��������˫����ֱ�ߵĻҶ�ֵ��
		int both_side_gray_diff_j=0;
		int both_side_gray_diff_i=0;
		float both_side_gray_diff_pro_i=0;
		float both_side_gray_diff_pro_j=0;

		for(unsigned int k=0;k<sto_linealong_gray.size();k++)
		{
		if(sto_linealong_gray[k]>sto_linealong_gray_i_5[k])
			both_side_gray_diff_i++;
		if(sto_linealong_gray[k]>sto_linealong_gray_j_5[k])
			both_side_gray_diff_j++;
		}
		both_side_gray_diff_pro_i=(float)both_side_gray_diff_i/(float)sto_linealong_gray.size();
		both_side_gray_diff_pro_j=(float)both_side_gray_diff_j/(float)sto_linealong_gray.size();

#ifdef REPORT
		cout<<"/*--------------�Ҷ��������------------*/"<<"\n";
		cout<<"����5�߶λҶȾ�ֵ"<<line_mean_gray_i_5<<"\n";
		cout<<"����1�߶λҶȾ�ֵ"<<line_mean_gray_i_1<<"\n";
		cout<<"�߶λҶȾ�ֵ"<<line_mean_gray<<"\n";
		cout<<"����1�߶λҶȾ�ֵ"<<line_mean_gray_j_1<<"\n";
		cout<<"����5�߶λҶȾ�ֵ"<<line_mean_gray_j_5<<"\n";
		cout<<"����5�߶����ҶȱȽϱ���"<<both_side_gray_diff_pro_i<<"\n";
		cout<<"����5�߶����ҶȱȽϱ���"<<both_side_gray_diff_pro_j<<"\n";
#endif

		//��һϵ��׼�����жϻҶ�flag���統ԭʼֱ���ϱ�˫��ֱ�߻Ҷȴ�����ص�ռֱ�����������ص��60%ʱ�жϻҶ�flagΪ��
		int gray_flag=0;
		decide_gray(line_mean_gray,
					line_mean_gray_i_1,line_mean_gray_j_1,
					line_mean_gray_i_5,line_mean_gray_j_5,
					both_side_gray_diff_pro_i,both_side_gray_diff_pro_j,&gray_flag);

		//���Ҷ�flag���ݶ�flag��Ϊ��ʱ����Ϊֱ������ȷ�Ļ���
		if(gray_flag==1 && grad_flag==1)
		{corret_id->push_back(i);
		}

#ifdef LINE_DEBUG

		//�������ڽ����޳���ֱ��
		Mat fiber_drawline;
		fiber_hat.copyTo(fiber_drawline);
		for(int jj=0;jj<line_side[i].size();jj++)
			fiber_drawline.at<uchar>(line_side[i][jj].y,line_side[i][jj].x)=255;
		imshow("fiber_drawline",fiber_drawline);
		waitKey();

#endif
	}

	return 0;
}

int fiber_scratch_finder(Mat& fiber_img_gray,Point center,int radius,int* scratch__amount)
{
	Rect rect_roi;
	rect_roi.x=center.x-radius;
	rect_roi.y=center.y-radius;
	rect_roi.width=2*radius;
	rect_roi.height=2*radius;;
	Mat fiber_img_scratch_img_src=Mat::Mat(fiber_img_gray,rect_roi);
	Mat fiber_img_scratch_img;

	Mat fiber_bkbk;
	fiber_img_scratch_img_src.copyTo(fiber_bkbk);//����ԭͼ���Ա������ԭͼ�ϱ���ҵ���ֱ��

	/*----------ͼ����ǿ��������NLM�˲������жԱȶȵ�����-----------------*///����fiber_img_scratch_img_src���fiber_img_scratch_img
	GaussianBlur(fiber_img_scratch_img_src,fiber_img_scratch_img,Size(3,3),0.5);
	nonlocalmeans_Proc(fiber_img_scratch_img);//nlm�˲�����ʵ�����ø�˹�˲���������nlm�˲�������������retinexͼ����ǿ�����㷨��ž����ø�˹ģ������ͼ����ģ����գ���Ϊ������+���屾��Ҷ�=ͼ��������ĻҶȡ��ɴ���������屾��ĻҶ�
	medianBlur(fiber_img_scratch_img,fiber_img_scratch_img,3);

	/*----------���ڻҶȷ�Χ����������-----------*///����fiber_img_scratch_img���fiber_growgray
	Mat fiber_growgray=Mat::Mat(fiber_img_scratch_img.rows,fiber_img_scratch_img.cols,fiber_img_scratch_img.type());
	region_grow_basegray(fiber_img_scratch_img,&fiber_growgray);//���ڻҶȷ�Χ����������//��180���ϵ������ҳ�������Щ�������ذ������ڻҶȷ�Χ��20���ڵ����غϲ�Ϊһ����ͨ��

	int fiber_noise_amount=0;
	vector<vector<Point>> fiber_noise_unidom;
	Mat fiber_growgray_grow2;
	fiber_growgray.copyTo(fiber_growgray_grow2);
	region_grow_2(fiber_growgray_grow2,&fiber_noise_amount,&fiber_noise_unidom);//������������fiber_growgray_grow2�е���ͨ���������fiber_noise_unidom
	fiber_growgray_grow2.release();
	cull_tinyregion(fiber_noise_unidom,fiber_growgray,30);//��size��30���µ���ͨ���޳�
	fiber_noise_unidom.clear();

	/*--------ϸ��-----*///����fiber__growgray���fiber_thin_inv
	Mat fiber_thin=Mat::Mat(fiber_growgray.rows,fiber_growgray.cols,fiber_growgray.type());
	fastThin(fiber_growgray,&fiber_thin,THIN_ITER);//ϸ����ͨ��
	fiber_growgray.release();

	//������Բ�����ͼ�񲿷��г�����Ϊ�ڣ�
	Mat fiber_thin_mask=Mat::Mat(fiber_thin.rows,fiber_thin.cols,fiber_thin.type(),Scalar(255));
	circle(fiber_thin_mask,Point(radius,radius),radius-5,Scalar(0),-1);
	fiber_thin.setTo(Scalar(0),fiber_thin_mask);

	vector<vector<Point>> fiber_thin_unidom;
	Mat fiber_thin_inv;
	bitwise_not(fiber_thin,fiber_thin_inv);
	int thin_region_amount=0;
	Mat fiber_thin_inv_grow;
	fiber_thin_inv.copyTo(fiber_thin_inv_grow);
	region_grow_2(fiber_thin_inv_grow,&thin_region_amount,&fiber_thin_unidom);//������������ϸ�������ͨ�����fiber_thin_unidom
	fiber_thin_inv_grow.release();
	cull_tinyregion(fiber_thin_unidom,fiber_thin_inv,4);//��size��4���µ���ͨ���޳�

	/*----------��houghlineP������ֱ�ߣ���ֵ��úܵͣ�-----------*///����fiber_thin_inv
	Mat fiber_thin_canny;
	Canny(fiber_thin_inv, fiber_thin_canny, 50, 200, 3 );
	vector<Vec4i> lines;
	HoughLinesP(fiber_thin_canny, lines, 1, CV_PI/180, HOUGH_TH, 135,50 );


	/*--------�������޳����ֱ�ߵĲ��֣��ⲿ�ִ�����fiber_scratchfinder���̵ļ򻯰棬Ч����������--------*/
	//�㷨��˼·�����ҵ���ֱ���������ֱ��������߶Σ�����Щ�߶εĻҶȺ��ݶ���ԭʼֱ�ߣ�λ���м�λ�ã����бȽ�

	//houghlineP�ҵ���ֱ�ߴ�������ǰ��������ϸߣ�Խ����ֱ������Խ����Կ����޶�ȡǰ�漸��ֱ�ߡ�
	int line_size_refine;
	if(lines.size()>LINES_NUM_ALLOW)
		line_size_refine=LINES_NUM_ALLOW;
	else line_size_refine=lines.size();

	//����ֱ�����ڴ洢����ԭʼֱ�������ֱ��
		vector<vector<Point>> line_side(line_size_refine),line_side_i_1(line_size_refine),
	line_side_i_2(line_size_refine),line_side_i_3(line_size_refine),line_side_i_4(line_size_refine),
	line_side_i_5(line_size_refine),line_side_j_1(line_size_refine),line_side_j_2(line_size_refine),
	line_side_j_3(line_size_refine),line_side_j_4(line_size_refine),line_side_j_5(line_size_refine);
	//1��������߶Σ�5����Զ���߶�
	//����houghlineP�Ľ��lines������ֱ�ߵĵ��������
	line_pt_generate(lines,line_size_refine,line_side,line_side_i_1,line_side_i_2,line_side_i_3,
		line_side_i_4,line_side_i_5,line_side_j_1,line_side_j_2,line_side_j_3,
		line_side_j_4,line_side_j_5);
	
	img_enhance(&fiber_img_scratch_img_src);//��ñ��ñ�任��С����ǿԭͼ�ĶԱȶȣ�ԭʼֱ�ߺ�������ֱ�ߵıȽ��ڸ���ǿͼ�Ͻ���
	Mat sto_grad_map=Mat::Mat(fiber_img_scratch_img_src.rows,fiber_img_scratch_img_src.cols,CV_32FC1);
	img_getsobelmap(fiber_img_scratch_img_src,&sto_grad_map);//��sobel�ݶȣ�ÿ�����ص�sobel�ݶ�ֵ����sto_grad_map��

	//�޳�����ֱ��
	vector<int> correct_line_id;
	cull_lines(fiber_img_scratch_img_src,line_side,line_side_i_1,line_side_i_2,line_side_i_3,
		line_side_i_4,line_side_i_5,line_side_j_1,line_side_j_2,line_side_j_3,
		line_side_j_4,line_side_j_5,sto_grad_map,&correct_line_id);


	/*--------ֱ�߼�⡢����ϣ��������м�⵽��ֱ��-----------*/
	Mat fiber_draw_scratch;
	cvtColor(fiber_bkbk,fiber_draw_scratch,CV_GRAY2BGR);
	for(unsigned int i=0;i<correct_line_id.size();i++)
		for(unsigned int j=0;j<line_side[correct_line_id[i]].size();j++)
		{
		 fiber_draw_scratch.at<uchar>(line_side[correct_line_id[i]][j].y,line_side[correct_line_id[i]][j].x*3)=0;
		 fiber_draw_scratch.at<uchar>(line_side[correct_line_id[i]][j].y,line_side[correct_line_id[i]][j].x*3+1)=0;
		 fiber_draw_scratch.at<uchar>(line_side[correct_line_id[i]][j].y,line_side[correct_line_id[i]][j].x*3+2)=255;
		}
		*scratch__amount=correct_line_id.size();
		printf("��⵽���۹�%d��\n",*scratch__amount);
		imshow("fiber_draw_scratch",fiber_draw_scratch);
		waitKey();
	
	return 0;
}
