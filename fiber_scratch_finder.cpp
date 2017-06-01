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
				double low=0.6645, double high=0.9332,   // X方向：low and high are the intensities of src
				double bottom=0, double top=1, // Y方向：mapped to bottom and top of dst
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




	//float转float
	if(flag==0){
		for(int i=0;i<src.rows;i++)
			for(int j=0;j<src.cols;j++)
			{
				double stosrc=(src.at<float>(i,j)-min_val)/(max_val-min_val);
				src.at<float>(i,j)=(float)stosrc;
			}
	}

	//uchar转uchar，覆盖原图片
	else if(flag==1){
		for(int i=0;i<src.rows;i++)
			for(int j=0;j<src.cols;j++)
			{
				double stosrc=(src.at<uchar>(i,j)-min_val)/(max_val-min_val)*255.0;
				src.at<uchar>(i,j)=(uchar)(stosrc+0.5);
			}
	}

	//float转uchar
	if(flag==2){
		for(int i=0;i<src.rows;i++)
			for(int j=0;j<src.cols;j++)
			{
				double stosrc=(src.at<float>(i,j)-min_val)/(max_val-min_val)*255.0;
				(*dst).at<uchar>(i,j)=int(stosrc+0.5);
			}
	}


	//int转float
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

	//下面一段程序利用插值的办法计算gv1和gv2，但暂没实现，先用matlab的参数代替
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

	int con=1;//用于判别是哪一类型的情况，con=1为斜率在1-0间
	if(x1>x2)//使两坐标为左到右排列
	{
		int r=x1;x1=x2;x2=r;
		r=y1;y1=y2;y2=r;
	}
	if(y2<y1)
	{
		y1=-y1,y2=-y2; //使这关于x作对称变换
		if(fabs(double(y2-y1))>fabs(double(x1-x2)))//斜率为小于-1 
		{
			con=4;
			int r=x1;x1=y1;y1=r;
			r=x2;x2=y2;y2=r;
		}
		else //斜率在-1到0间
			con=3;
	}
	else if(y2-y1>x2-x1)//斜率大于1 使之关于y=x对称变换
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
		switch(con)//还原为适当坐标
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


	//BresenhamLine函数可以根据图像上直线的两个端点得到该直线上所有的点
	//用bh画出这以上线段，并找到线段上的点
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


		//判断线段倾向x方向还是y方向x1-x2>y1-y2,x是列y是行
		if(abs(lines[i][0]-lines[i][2])>abs(lines[i][1]-lines[i][3]))//x>y，直线趋于水平，向y轴移动得到两侧的直线
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


	/*-------求梯度最值--------*/
	float max_line_grad=-1;
	float min_line_grad=99999;
	for(int i=0;i<11;i++)
	{
	if(g[i]>=max_line_grad)
		max_line_grad=g[i];
	if(g[i]<min_line_grad)
		min_line_grad=g[i];
	}


	/*------找到处于中间部份的梯度最值，一般处于中间线段的两侧，即找到line4,line5,line6这三条线的哪一条的梯度值最大--------*/
	float middle_max_grad=-1;
	int middle_max_grad_id=0;
	for(int i=0;i<3;i++)
		if(g[4+i]>=middle_max_grad)
		{
		   middle_max_grad_id=4+i;
		   middle_max_grad=g[4+i];
		}
	

	/*--------判断在这含原始直线的10条直上，梯度的波动是否符合双峰的分布，---------*/
	//先求均值和标准差
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

	//统计梯度分布的波峰、波谷的变化情况
	for(int i=0;i<(11-1);i++)
	{
		if(g[i+1]-g[i]>=0)
		cur_sign=1;
		else cur_sign=-1;

		if(cur_sign!=pre_sign)//在这里有一个波谷或波峰
			sign_change++;

		pre_sign=cur_sign;
	}


	//如果梯度分布的波峰、波谷的变化等于3次，说明梯度分布符合双峰分布，认为直线是真正的划痕（我们认为直划痕在划痕痕及划痕双侧的梯度变化符合双峰分布）
	//梯度最大最小值超过5，说明梯度变化较大，在梯度分布符合双峰分布的情况下判断为真正的划痕
	  if(sign_change==3)
		if(abs(max_line_grad-min_line_grad)>5)
		{
			*flag=1;
			//printf("检测到双峰,梯度信号置为真");	
		}

		//梯度最大最小值超过8，说明梯度变化很大，是真正的划痕
		if(abs(max_line_grad-min_line_grad)>8)
		{
			*flag=1;
		//	printf("梯度信号置为真\n");
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

		if(max_gray-line_mean_gray_i_5>1 && max_gray-line_mean_gray_j_5>1)//和距离最远的线灰度均值进行比较
			if(max_gray*2-line_mean_gray_i_5-line_mean_gray_j_5>2)
			{
				*gray_flag=1;
				//printf("通过灰度信号检测器1\n");
			}	
	}

	if(both_side_gray_diff_pro_i>0.5 && both_side_gray_diff_pro_j>0.5 ||
		(both_side_gray_diff_pro_i>0.8 || both_side_gray_diff_pro_j>0.8))
		if(max_gray-line_mean_gray_i_1>1 && max_gray-line_mean_gray_i_1>1)//和距离最近的线灰度均值进行比较
			if(max_gray*2-line_mean_gray_i_1-line_mean_gray_j_1>2)
			{
				*gray_flag=1;
				//printf("通过灰度信号检测器2\n");
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

	//遍历每一条直线
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
		//遍历每一条直线上的每一个点
		for(unsigned int j=0;j<line_side[i].size();j++)
		{

			//从fiber_grad_map中取出每条直线的梯度值
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

			//从fiber_hat中取出每条直线的灰度值
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
		cout<<"/*--------------梯度值情况报告------------*/"<<"\n";
		cout<<"距离5线段的梯度总值是"<<sto_line_grad_i_5/(double)line_side_i_5[i].size()<<"\n";
		cout<<"距离4线段的梯度总值是"<<sto_line_grad_i_4/(double)line_side_i_4[i].size()<<"\n";
		cout<<"距离3线段的梯度总值是"<<sto_line_grad_i_3/(double)line_side_i_3[i].size()<<"\n";
		cout<<"距离2线段的梯度总值是"<<sto_line_grad_i_2/(double)line_side_i_2[i].size()<<"\n";
		cout<<"距离1线段的梯度总值是"<<sto_line_grad_i_1/(double)line_side_i_1[i].size()<<"\n";
		cout<<"线段的梯度总值是"<<sto_line_grad/(double)line_side[i].size()<<"\n";
		cout<<"距离1线段的梯度总值是"<<sto_line_grad_j_1/(double)line_side_j_1[i].size()<<"\n";
		cout<<"距离2线段的梯度总值是"<<sto_line_grad_j_2/(double)line_side_j_2[i].size()<<"\n";
		cout<<"距离3线段的梯度总值是"<<sto_line_grad_j_3/(double)line_side_j_3[i].size()<<"\n";
		cout<<"距离4线段的梯度总值是"<<sto_line_grad_j_4/(double)line_side_j_4[i].size()<<"\n";
		cout<<"距离5线段的梯度总值是"<<sto_line_grad_j_5/(double)line_side_j_5[i].size()<<"\n";
#endif

		/*--------通过原始直线两侧的梯度值来判断直线的质量-----*/
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

		/*--------通过原始直线两侧的灰度值来判断直线的质量-----*/
		//计算线段上的平均灰度值，及双侧距离最远的直线的灰度均值
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

		//对比原始线段上每一点的灰度值和双侧(矩离最远的)线段上灰度值（我们认为理想情况下划痕上每一点的灰度值都大于其双侧上直线的灰度值）
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
		cout<<"/*--------------灰度情况报告------------*/"<<"\n";
		cout<<"距离5线段灰度均值"<<line_mean_gray_i_5<<"\n";
		cout<<"距离1线段灰度均值"<<line_mean_gray_i_1<<"\n";
		cout<<"线段灰度均值"<<line_mean_gray<<"\n";
		cout<<"距离1线段灰度均值"<<line_mean_gray_j_1<<"\n";
		cout<<"距离5线段灰度均值"<<line_mean_gray_j_5<<"\n";
		cout<<"距离5线段逐点灰度比较比例"<<both_side_gray_diff_pro_i<<"\n";
		cout<<"距离5线段逐点灰度比较比例"<<both_side_gray_diff_pro_j<<"\n";
#endif

		//用一系列准则来判断灰度flag，如当原始直线上比双侧直线灰度大的像素点占直线上所有像素点的60%时判断灰度flag为真
		int gray_flag=0;
		decide_gray(line_mean_gray,
					line_mean_gray_i_1,line_mean_gray_j_1,
					line_mean_gray_i_5,line_mean_gray_j_5,
					both_side_gray_diff_pro_i,both_side_gray_diff_pro_j,&gray_flag);

		//当灰度flag和梯度flag都为真时，认为直线是正确的划痕
		if(gray_flag==1 && grad_flag==1)
		{corret_id->push_back(i);
		}

#ifdef LINE_DEBUG

		//画出正在进行剔除的直线
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
	fiber_img_scratch_img_src.copyTo(fiber_bkbk);//备份原图，以便后面在原图上标记找到的直线

	/*----------图像增强，采用了NLM滤波器进行对比度的拉伸-----------------*///输入fiber_img_scratch_img_src输出fiber_img_scratch_img
	GaussianBlur(fiber_img_scratch_img_src,fiber_img_scratch_img,Size(3,3),0.5);
	nonlocalmeans_Proc(fiber_img_scratch_img);//nlm滤波，其实这里用高斯滤波器代替了nlm滤波器，本质上是retinex图像增强法，算法大概就是用高斯模糊掉的图像来模拟光照，认为：光照+物体本身灰度=图像上物体的灰度。由此来求出物体本身的灰度
	medianBlur(fiber_img_scratch_img,fiber_img_scratch_img,3);

	/*----------基于灰度范围的区域生长-----------*///输入fiber_img_scratch_img输出fiber_growgray
	Mat fiber_growgray=Mat::Mat(fiber_img_scratch_img.rows,fiber_img_scratch_img.cols,fiber_img_scratch_img.type());
	region_grow_basegray(fiber_img_scratch_img,&fiber_growgray);//基于灰度范围的区域生长//将180以上的像素找出，在这些高亮像素八邻域内灰度范围是20以内的像素合并为一个联通域

	int fiber_noise_amount=0;
	vector<vector<Point>> fiber_noise_unidom;
	Mat fiber_growgray_grow2;
	fiber_growgray.copyTo(fiber_growgray_grow2);
	region_grow_2(fiber_growgray_grow2,&fiber_noise_amount,&fiber_noise_unidom);//区域生长，将fiber_growgray_grow2中的联通域存于容器fiber_noise_unidom
	fiber_growgray_grow2.release();
	cull_tinyregion(fiber_noise_unidom,fiber_growgray,30);//将size在30以下的联通域剔除
	fiber_noise_unidom.clear();

	/*--------细化-----*///输入fiber__growgray输出fiber_thin_inv
	Mat fiber_thin=Mat::Mat(fiber_growgray.rows,fiber_growgray.cols,fiber_growgray.type());
	fastThin(fiber_growgray,&fiber_thin,THIN_ITER);//细化联通域
	fiber_growgray.release();

	//将光纤圆面外的图像部份切除（设为黑）
	Mat fiber_thin_mask=Mat::Mat(fiber_thin.rows,fiber_thin.cols,fiber_thin.type(),Scalar(255));
	circle(fiber_thin_mask,Point(radius,radius),radius-5,Scalar(0),-1);
	fiber_thin.setTo(Scalar(0),fiber_thin_mask);

	vector<vector<Point>> fiber_thin_unidom;
	Mat fiber_thin_inv;
	bitwise_not(fiber_thin,fiber_thin_inv);
	int thin_region_amount=0;
	Mat fiber_thin_inv_grow;
	fiber_thin_inv.copyTo(fiber_thin_inv_grow);
	region_grow_2(fiber_thin_inv_grow,&thin_region_amount,&fiber_thin_unidom);//区域生长，将细化后的联通域存于fiber_thin_unidom
	fiber_thin_inv_grow.release();
	cull_tinyregion(fiber_thin_unidom,fiber_thin_inv,4);//将size在4以下的联通域剔除

	/*----------用houghlineP找所有直线（阈值设得很低）-----------*///输入fiber_thin_inv
	Mat fiber_thin_canny;
	Canny(fiber_thin_inv, fiber_thin_canny, 50, 200, 3 );
	vector<Vec4i> lines;
	HoughLinesP(fiber_thin_canny, lines, 1, CV_PI/180, HOUGH_TH, 135,50 );


	/*--------下面是剔除误检直线的部分，这部分代码是fiber_scratchfinder工程的简化版，效果有所不及--------*/
	//算法的思路是在找到的直线两侧作分别作五条线段，将这些线段的灰度和梯度与原始直线（位于中间位置）进行比较

	//houghlineP找到的直线存在容器前面的质量较高，越往后直线质量越差，所以可以限定取前面几条直线。
	int line_size_refine;
	if(lines.size()>LINES_NUM_ALLOW)
		line_size_refine=LINES_NUM_ALLOW;
	else line_size_refine=lines.size();

	//建立直线用于存储所有原始直线两侧的直线
		vector<vector<Point>> line_side(line_size_refine),line_side_i_1(line_size_refine),
	line_side_i_2(line_size_refine),line_side_i_3(line_size_refine),line_side_i_4(line_size_refine),
	line_side_i_5(line_size_refine),line_side_j_1(line_size_refine),line_side_j_2(line_size_refine),
	line_side_j_3(line_size_refine),line_side_j_4(line_size_refine),line_side_j_5(line_size_refine);
	//1是最近的线段，5是最远的线段
	//根据houghlineP的结果lines将所有直线的点存入容器
	line_pt_generate(lines,line_size_refine,line_side,line_side_i_1,line_side_i_2,line_side_i_3,
		line_side_i_4,line_side_i_5,line_side_j_1,line_side_j_2,line_side_j_3,
		line_side_j_4,line_side_j_5);
	
	img_enhance(&fiber_img_scratch_img_src);//顶帽底帽变换，小幅增强原图的对比度，原始直线和其两侧直线的比较在该增强图上进行
	Mat sto_grad_map=Mat::Mat(fiber_img_scratch_img_src.rows,fiber_img_scratch_img_src.cols,CV_32FC1);
	img_getsobelmap(fiber_img_scratch_img_src,&sto_grad_map);//求sobel梯度，每个像素的sobel梯度值存在sto_grad_map中

	//剔除误检的直线
	vector<int> correct_line_id;
	cull_lines(fiber_img_scratch_img_src,line_side,line_side_i_1,line_side_i_2,line_side_i_3,
		line_side_i_4,line_side_i_5,line_side_j_1,line_side_j_2,line_side_j_3,
		line_side_j_4,line_side_j_5,sto_grad_map,&correct_line_id);


	/*--------直线检测、求精完毕，画出所有检测到的直线-----------*/
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
		printf("检测到划痕共%d条\n",*scratch__amount);
		imshow("fiber_draw_scratch",fiber_draw_scratch);
		waitKey();
	
	return 0;
}
