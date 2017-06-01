#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
#include "img_region_grow.h"
#define  CLOSE_SIZE 3
#define PAI 3.14159265358979
#define CENTER_FINDER_ITER 4

inline bool fiberPointSizeGreater(vector<Point>& lhs, vector<Point>& rhs)
{
	return lhs.size() > rhs.size();
}

int fiber_rot_img(Mat src_mat, Mat* rot_mat, int rot_affin_angle)
{
	float affin_m[6];
	CvMat affin_M = Mat(2, 3, CV_32F, affin_m);
	int affin_w = src_mat.cols;
	int affin_h = src_mat.rows;

	affin_m[0] = (float)(cos(-rot_affin_angle  * CV_PI / 180.));
	affin_m[1] = (float)(sin(-rot_affin_angle  * CV_PI / 180.));
	affin_m[3] = -affin_m[1];
	affin_m[4] = affin_m[0];
	// 将旋转中心移至图像中间
	affin_m[2] = affin_w * 0.5f;
	affin_m[5] = affin_h * 0.5f;

	CvMat affin_src = src_mat;
	CvMat affin_dst = *rot_mat;
	cvGetQuadrangleSubPix(&affin_src, &affin_dst, &affin_M);

	return 0;
}

int adjust_rot_point(int rot_affin_angle, int rot_center_x, int rot_center_y, float* sav_center_x, float *sav_center_y)
{
	double center_distance = (*sav_center_x - rot_center_x)*(*sav_center_x - rot_center_x) + (*sav_center_y - rot_center_y)*(*sav_center_y - rot_center_y);
	center_distance = sqrt(center_distance);
	double tmp_L1 = atan((double)(*sav_center_y - rot_center_y) / (double)(*sav_center_x - rot_center_x));
	tmp_L1 = abs(tmp_L1);
	double rot_affin_angle_rad = rot_affin_angle*PAI / 180.0;//（角度转弧度）
	//分象限讨论
	if (*sav_center_y - rot_center_y > 0 && *sav_center_x - rot_center_x > 0)//第一象限
		tmp_L1 += rot_affin_angle_rad;
	else if (*sav_center_y - rot_center_y > 0 && *sav_center_x - rot_center_x < 0)//第二象限
		tmp_L1 = PAI - tmp_L1 + rot_affin_angle_rad;
	else if (*sav_center_y - rot_center_y < 0 && *sav_center_x - rot_center_x < 0)//第三象限
		tmp_L1 = PAI + tmp_L1 + rot_affin_angle_rad;
	else if (*sav_center_y - rot_center_y < 0 && *sav_center_x - rot_center_x>0)//第四象限
		tmp_L1 = 2 * PAI - tmp_L1 + rot_affin_angle_rad;
	//求映射
	double tmp_y_proj = (double)sin(tmp_L1)*center_distance;
	double tmp_x_proj = (double)cos(tmp_L1)*center_distance;
	tmp_x_proj = rot_center_x + tmp_x_proj;
	tmp_y_proj = rot_center_y + tmp_y_proj;
	*sav_center_x = (float)tmp_x_proj;
	*sav_center_y = (float)tmp_y_proj;

	return 0;
}
int find_circle_centers(Mat rot_square, vector<Point2f>* probb_centers, short* probb_radius)
{
	int sav_min_r = rot_square.rows;
	/*-------将原图以45度为单位旋转一周------*/
	for (int rot_affin_angle = 0; rot_affin_angle < 360; rot_affin_angle = rot_affin_angle + 45)
	{
		Mat rot_img = Mat::Mat(rot_square.rows, rot_square.cols, rot_square.type(), Scalar(0));

		fiber_rot_img(rot_square, &rot_img, rot_affin_angle);//旋转

		vector<Point> ang_proj_v(rot_square.cols);
		vector<Point> ang_proj_h(rot_square.rows);
		vector<Point> proj_contours;
		float sav_center_x = -1;
		float sav_center_y = -1;

		/*--------作垂直投影-------*///
		for (int j = 0; j < rot_img.cols; j++)
		{
			for (int i = 0; i < rot_img.rows; i++)
				ang_proj_v[j].y += rot_img.at<uchar>(i, j) / 255;//计算垂直投影，圆形的投景是抛物线//除以255是为了将二值化后的图归一化
			if (ang_proj_v[j].y != 0)//作垂直的镜像，保存镜像的像素点
			{
				proj_contours.push_back(Point(j, ang_proj_v[j].y));
				proj_contours.push_back(Point(j, -ang_proj_v[j].y));
			}
		}
		RotatedRect v_elps = fitEllipse(proj_contours);//将投影图的轮廓点进行椭圆拟合
		sav_center_x = v_elps.center.x;//取出椭圆的圆心（x的中点）
		if (sav_min_r > v_elps.size.width)
			sav_min_r = int(v_elps.size.width + 0.5);//若拟合的的椭圆更小，则更新圆心（我们认为越小的椭圆越好）
		proj_contours.clear();

		/*--------作水平投影-------*///
		for (int i = 0; i < rot_img.rows; i++) {
			for (int j = 0; j < rot_img.cols; j++)
			{
				ang_proj_h[i].x += rot_img.at<uchar>(i, j) / 255;//除以255是为了将二值化后的图归一化
			}
			if (ang_proj_h[i].x != 0)
			{
				proj_contours.push_back(Point(ang_proj_h[i].x, i));
				proj_contours.push_back(Point(-ang_proj_h[i].x, i));
			}
		}
		RotatedRect h_elps = fitEllipse(proj_contours);
		sav_center_y = h_elps.center.y;
		if (sav_min_r > h_elps.size.width)
			sav_min_r = int(h_elps.size.width + 0.5);
		proj_contours.clear();

		//校正得到的圆心坐标，因旋转后进行投影而得到的投影中心并不是原始的圆心位置，必须将其校正回原来的坐标位置
		adjust_rot_point(rot_affin_angle, rot_square.cols / 2, rot_square.rows / 2, &sav_center_x, &sav_center_y);
		(*probb_centers).push_back(Point2f(sav_center_x, sav_center_y));
	}
	*probb_radius = (short)(sav_min_r / 2 + 0.5);//将上述步骤得出的最小半径进行保存
	return 0;
}

int ad_center_point(int roi_width, int roi_height, int suare_width, vector<Point2f>* probb_centers)
{
	/*----------求所有圆心的均值---------------*/
	Point2f mean_center(-1, -1);
	for (unsigned int i = 0; i < (*probb_centers).size(); i++)
	{
		mean_center.x += (*probb_centers)[i].x;
		mean_center.y += (*probb_centers)[i].y;
	}
	mean_center.x /= (*probb_centers).size();
	mean_center.y /= (*probb_centers).size();

	/*---------求所有圆心的标准差-----------*/
	Point2f stand_devia = -1;
	for (unsigned int i = 0; i < (*probb_centers).size(); i++)
	{
		stand_devia.x += ((*probb_centers)[i].x - mean_center.x)*((*probb_centers)[i].x - mean_center.x);
		stand_devia.y += ((*probb_centers)[i].y - mean_center.y)*((*probb_centers)[i].y - mean_center.y);
	}

	stand_devia.x /= (*probb_centers).size();
	stand_devia.y /= (*probb_centers).size();
	stand_devia.x = sqrt(stand_devia.x);
	stand_devia.y /= sqrt(stand_devia.y);

	/*---------剔除噪声圆心------*/
	for (unsigned int i = 0; i < (*probb_centers).size(); i++)
	{
		//将偏离均值4倍标准差的点剔除
		if (abs((*probb_centers)[i].x - mean_center.x) > 4 * stand_devia.x || abs((*probb_centers)[i].y - mean_center.y) > 4 * stand_devia.y)
		{
			probb_centers->erase(probb_centers->begin() + i);
			i--;
		}
		else
		{
			(*probb_centers)[i].x -= (suare_width - roi_width) / 2;//转换成原图的坐标
			(*probb_centers)[i].y -= (suare_width - roi_height) / 2;
			(*probb_centers)[i].x = (float)(int)((*probb_centers)[i].x + 0.5);
			(*probb_centers)[i].y = (float)(int)((*probb_centers)[i].y + 0.5);
		}
	}
	return 0;
}


bool findcenterpoinxmax(Point2f& lhs, Point2f& rhs)
{
	return lhs.x < rhs.x;
}
bool findcenterpoinymax(Point2f& lhs, Point2f& rhs)
{
	return lhs.y < rhs.y;
}

int find_best_center(Mat draw_max_unidom, vector<Point2f> probb_centers, short radius, Point* best_point, short* best_radius, int* eva)
{

	/*---------求出所有候选圆心坐标上的最值-----------*/
	float center_x_max = std::max_element(probb_centers.begin(), probb_centers.end(), findcenterpoinxmax)->x;
	float center_x_min = std::min_element(probb_centers.begin(), probb_centers.end(), findcenterpoinxmax)->x;
	float center_y_max = std::max_element(probb_centers.begin(), probb_centers.end(), findcenterpoinymax)->y;
	float center_y_min = std::min_element(probb_centers.begin(), probb_centers.end(), findcenterpoinymax)->y;

	radius++;//在大多数情况下半径总是偏小所以+1，以便以下的圆心的确率判决

	//计算圆的最大外接矩形，圆必在其内
	int  encls_rec_x_min = int(center_x_min - radius + 0.5);
	int  encls_rec_y_min = int(center_y_min - radius + 0.5);
	int  encls_rec_x_max = int(center_x_max + radius + 0.5);
	int  encls_rec_y_max = int(center_y_max + radius + 0.5);

	if (encls_rec_x_min < 0 || encls_rec_y_min < 0 || (encls_rec_x_max - encls_rec_x_min) < 0 || (encls_rec_y_max - encls_rec_y_min) < 0)
	{
		printf("圆心定位失败，无法找到圆心所在范围");
		*eva |= 0x02;
		return -1;
	}

	int min_frag = (encls_rec_x_max - encls_rec_x_min)*(encls_rec_y_max - encls_rec_y_min);
	int sav_best_center_x = -1;
	int sav_best_center_y = -1;


	/*-------------------遍历所有候选圆心，分别在光纤圆面（光纤白，背景黑）图上画黑色的圆，看绘制的圆和光纤圆面不重合的部分（即白色部分）有多少，以此来确定最好的圆心--------*/

	for (unsigned int i = 0; i < probb_centers.size(); i++)
	{
		Mat fill_circle_black;
		draw_max_unidom.copyTo(fill_circle_black);//光纤圆面的图//背景是黑色，光纤圆面是白色
		Point probb_centers_int((int)(probb_centers[i].x + 0.5), (int)(probb_centers[i].y + 0.5));//取出候选圆心
		circle(fill_circle_black, probb_centers_int, radius, Scalar(0), -1);//以候选圆心为圆心画黑色的圆
		//imshow("fill_circle_black",fill_circle_black);
		//waitKey();

		//求出用绘制的圆填充光纤圆面后，剩余的白色部分有多少（越少说明绘制的圆越接近真实的光纤圆面）
		int frag = 0;
		for (int j = encls_rec_y_min; j < encls_rec_y_max; j++)
			for (int k = encls_rec_x_min; k < encls_rec_x_max; k++)
			{
				if (fill_circle_black.at<uchar>(j, k) == 255)
					frag++;
			}

		//存最小剩余部分的圆心坐标
		if (frag < min_frag)
		{
			min_frag = frag;
			sav_best_center_x = probb_centers_int.x;
			sav_best_center_y = probb_centers_int.y;
		}
	}

	/*--------求最佳圆心的8领域，绘制黑色圆去填充光纤圆面（背景黑，圆面白），看剩余部分多少，和上一个步骤类-----*/
	//遍历确定的圆心的8邻域，找到更好的圆心
	int x_8n[8] = { 0,0,-1,1,-1,1,-1,1 };
	int y_8n[8] = { -1,1,0,0,-1,-1,1,1 };

	double finder_iter = CENTER_FINDER_ITER;
	for (int k = 0; k < finder_iter; k++)
	{

		for (int n8_pt = 0; n8_pt < 8; n8_pt++)
		{
			Mat fill_circle_bk2;
			draw_max_unidom.copyTo(fill_circle_bk2);
			circle(fill_circle_bk2, Point(sav_best_center_x + x_8n[n8_pt], sav_best_center_y + y_8n[n8_pt]), radius, Scalar(0), -1);
			int frag = 0;
			for (int i = encls_rec_y_min; i < encls_rec_y_max; i++)
				for (int j = encls_rec_x_min; j < encls_rec_x_max; j++)
				{
					if (fill_circle_bk2.at<uchar>(i, j) == 255)
						frag++;
				}
			if (frag < min_frag)
			{
				min_frag = frag;
				sav_best_center_x = sav_best_center_x + x_8n[n8_pt];
				sav_best_center_y = sav_best_center_y + y_8n[n8_pt];
			}

		}
	}

	/*******************************
	缩放（放大半径，找到更好的半径）
	暂时没有办法实现
	********************************/
	//计算残余白色和圆外接矩形(扣除圆面积)的比例，若太大则定位失败
	float frag_percentage = (float)min_frag / (float)((encls_rec_x_max - encls_rec_x_min)*(encls_rec_y_max - encls_rec_y_min));
	if (frag_percentage >= 0.1)
	{
		printf("圆心定位失败，残余部份过大");
		*eva |= 0x02;
		return -1;
	}
	else
	{
		(*best_point).x = sav_best_center_x;
		(*best_point).y = sav_best_center_y;
		*best_radius = radius;
	}

	return 0;
}

int fiber_circle_locate(Mat fiber_img_gray, Point* circle_center, int* circle_radius, int* fiber_eva)
{
	//闭运算，将图中的杂质滤去或破坏
	Mat fiber_cls_element = getStructuringElement(MORPH_ELLIPSE,
		Size(2 * CLOSE_SIZE + 1, 2 * CLOSE_SIZE + 1),
		Point(CLOSE_SIZE, CLOSE_SIZE));
	morphologyEx(fiber_img_gray, fiber_img_gray, MORPH_CLOSE, fiber_cls_element, Point(-1, -1), 1);//
	Mat fiber_img_2val;
	//OTSU自适应阈值分割
	threshold(fiber_img_gray, fiber_img_2val, 0, 255, THRESH_OTSU);

	//找出最大的联通域，认为它就是圆
	vector<vector<Point>> fiber_unidom;
	int fiber_unidom_sum = 0;
	region_grow_2(fiber_img_2val, &fiber_unidom_sum, &fiber_unidom);//区域生长，将联通区域的点存入fiber_unidom
	nth_element(fiber_unidom.begin(), fiber_unidom.begin(), fiber_unidom.end(), fiberPointSizeGreater);//将联通区域按面积由大到小进行排序

	//画出最大的联通域，即光纤的圆面
	Mat draw_max_unidom = Mat::Mat(fiber_img_2val.rows, fiber_img_2val.cols, fiber_img_2val.type(), Scalar(0, 0, 0));
	for (unsigned int i = 0; i < fiber_unidom[0].size(); i++)//取已经排好顺序的最大面积联通域为圆，即取fiber_unidom[0]
		draw_max_unidom.at<uchar>(fiber_unidom[0][i].y, fiber_unidom[0][i].x) = 255;//在draw_max_unidom上绘出这个最大的联通区域（光纤圆面）

	/*-----------------旋转后求投影，寻找光纤圆心------------------------*/
	//做一个大的方形画布rot_square，将光纤画面fiber_sqr_roi_rec拷进方画布中央，以便将光纤圆面进行旋转和投影，而不致光纤圆面经旋转后超出画布
	int rot_square_r = (int)sqrt((float)(fiber_img_gray.rows*fiber_img_gray.rows + fiber_img_gray.cols*fiber_img_gray.cols));
	Mat rot_square = Mat::Mat(rot_square_r, rot_square_r, fiber_img_gray.type(), Scalar(0));
	Rect fiber_sqr_roi_rec;
	fiber_sqr_roi_rec.width = fiber_img_gray.cols;
	fiber_sqr_roi_rec.height = fiber_img_gray.rows;
	fiber_sqr_roi_rec.x = (rot_square_r - fiber_img_gray.cols) / 2;
	fiber_sqr_roi_rec.y = (rot_square_r - fiber_img_gray.rows) / 2;
	Mat fiber_sqr_roi;
	fiber_sqr_roi = Mat::Mat(rot_square, fiber_sqr_roi_rec);
	draw_max_unidom.copyTo(fiber_sqr_roi);
	vector<Point2f> probb_centers;//存所有可能的圆心
	short probb_radius;//存所有可能的半径
	find_circle_centers(rot_square, &probb_centers, &probb_radius);//旋转圆面，并做水平垂直投影，求出投影的中心点作为光纤圆心
	ad_center_point(fiber_sqr_roi_rec.width, fiber_sqr_roi_rec.height, rot_square.rows, &probb_centers);//剔除噪声圆心
	rot_square.release();

	/*--------精炼圆心的位置------------*/
	Point best_point(-1, -1);
	short best_radius = -1;
	int finder_complete = find_best_center(draw_max_unidom, probb_centers, probb_radius, &best_point, &best_radius, fiber_eva);//测试候选圆心和圆心的邻域像素点，进一步找到更佳的圆心
	if (finder_complete == -1)
		return -1;

	//存最佳的圆心和半径
	circle_center->x = best_point.x;
	circle_center->y = best_point.y;
	*circle_radius = (int)best_radius;

	return 0;
}