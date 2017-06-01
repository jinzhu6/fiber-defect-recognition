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
	// ����ת��������ͼ���м�
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
	double rot_affin_angle_rad = rot_affin_angle*PAI / 180.0;//���Ƕ�ת���ȣ�
	//����������
	if (*sav_center_y - rot_center_y > 0 && *sav_center_x - rot_center_x > 0)//��һ����
		tmp_L1 += rot_affin_angle_rad;
	else if (*sav_center_y - rot_center_y > 0 && *sav_center_x - rot_center_x < 0)//�ڶ�����
		tmp_L1 = PAI - tmp_L1 + rot_affin_angle_rad;
	else if (*sav_center_y - rot_center_y < 0 && *sav_center_x - rot_center_x < 0)//��������
		tmp_L1 = PAI + tmp_L1 + rot_affin_angle_rad;
	else if (*sav_center_y - rot_center_y < 0 && *sav_center_x - rot_center_x>0)//��������
		tmp_L1 = 2 * PAI - tmp_L1 + rot_affin_angle_rad;
	//��ӳ��
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
	/*-------��ԭͼ��45��Ϊ��λ��תһ��------*/
	for (int rot_affin_angle = 0; rot_affin_angle < 360; rot_affin_angle = rot_affin_angle + 45)
	{
		Mat rot_img = Mat::Mat(rot_square.rows, rot_square.cols, rot_square.type(), Scalar(0));

		fiber_rot_img(rot_square, &rot_img, rot_affin_angle);//��ת

		vector<Point> ang_proj_v(rot_square.cols);
		vector<Point> ang_proj_h(rot_square.rows);
		vector<Point> proj_contours;
		float sav_center_x = -1;
		float sav_center_y = -1;

		/*--------����ֱͶӰ-------*///
		for (int j = 0; j < rot_img.cols; j++)
		{
			for (int i = 0; i < rot_img.rows; i++)
				ang_proj_v[j].y += rot_img.at<uchar>(i, j) / 255;//���㴹ֱͶӰ��Բ�ε�Ͷ����������//����255��Ϊ�˽���ֵ�����ͼ��һ��
			if (ang_proj_v[j].y != 0)//����ֱ�ľ��񣬱��澵������ص�
			{
				proj_contours.push_back(Point(j, ang_proj_v[j].y));
				proj_contours.push_back(Point(j, -ang_proj_v[j].y));
			}
		}
		RotatedRect v_elps = fitEllipse(proj_contours);//��ͶӰͼ�������������Բ���
		sav_center_x = v_elps.center.x;//ȡ����Բ��Բ�ģ�x���е㣩
		if (sav_min_r > v_elps.size.width)
			sav_min_r = int(v_elps.size.width + 0.5);//����ϵĵ���Բ��С�������Բ�ģ�������ΪԽС����ԲԽ�ã�
		proj_contours.clear();

		/*--------��ˮƽͶӰ-------*///
		for (int i = 0; i < rot_img.rows; i++) {
			for (int j = 0; j < rot_img.cols; j++)
			{
				ang_proj_h[i].x += rot_img.at<uchar>(i, j) / 255;//����255��Ϊ�˽���ֵ�����ͼ��һ��
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

		//У���õ���Բ�����꣬����ת�����ͶӰ���õ���ͶӰ���Ĳ�����ԭʼ��Բ��λ�ã����뽫��У����ԭ��������λ��
		adjust_rot_point(rot_affin_angle, rot_square.cols / 2, rot_square.rows / 2, &sav_center_x, &sav_center_y);
		(*probb_centers).push_back(Point2f(sav_center_x, sav_center_y));
	}
	*probb_radius = (short)(sav_min_r / 2 + 0.5);//����������ó�����С�뾶���б���
	return 0;
}

int ad_center_point(int roi_width, int roi_height, int suare_width, vector<Point2f>* probb_centers)
{
	/*----------������Բ�ĵľ�ֵ---------------*/
	Point2f mean_center(-1, -1);
	for (unsigned int i = 0; i < (*probb_centers).size(); i++)
	{
		mean_center.x += (*probb_centers)[i].x;
		mean_center.y += (*probb_centers)[i].y;
	}
	mean_center.x /= (*probb_centers).size();
	mean_center.y /= (*probb_centers).size();

	/*---------������Բ�ĵı�׼��-----------*/
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

	/*---------�޳�����Բ��------*/
	for (unsigned int i = 0; i < (*probb_centers).size(); i++)
	{
		//��ƫ���ֵ4����׼��ĵ��޳�
		if (abs((*probb_centers)[i].x - mean_center.x) > 4 * stand_devia.x || abs((*probb_centers)[i].y - mean_center.y) > 4 * stand_devia.y)
		{
			probb_centers->erase(probb_centers->begin() + i);
			i--;
		}
		else
		{
			(*probb_centers)[i].x -= (suare_width - roi_width) / 2;//ת����ԭͼ������
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

	/*---------������к�ѡԲ�������ϵ���ֵ-----------*/
	float center_x_max = std::max_element(probb_centers.begin(), probb_centers.end(), findcenterpoinxmax)->x;
	float center_x_min = std::min_element(probb_centers.begin(), probb_centers.end(), findcenterpoinxmax)->x;
	float center_y_max = std::max_element(probb_centers.begin(), probb_centers.end(), findcenterpoinymax)->y;
	float center_y_min = std::min_element(probb_centers.begin(), probb_centers.end(), findcenterpoinymax)->y;

	radius++;//�ڴ��������°뾶����ƫС����+1���Ա����µ�Բ�ĵ�ȷ���о�

	//����Բ�������Ӿ��Σ�Բ��������
	int  encls_rec_x_min = int(center_x_min - radius + 0.5);
	int  encls_rec_y_min = int(center_y_min - radius + 0.5);
	int  encls_rec_x_max = int(center_x_max + radius + 0.5);
	int  encls_rec_y_max = int(center_y_max + radius + 0.5);

	if (encls_rec_x_min < 0 || encls_rec_y_min < 0 || (encls_rec_x_max - encls_rec_x_min) < 0 || (encls_rec_y_max - encls_rec_y_min) < 0)
	{
		printf("Բ�Ķ�λʧ�ܣ��޷��ҵ�Բ�����ڷ�Χ");
		*eva |= 0x02;
		return -1;
	}

	int min_frag = (encls_rec_x_max - encls_rec_x_min)*(encls_rec_y_max - encls_rec_y_min);
	int sav_best_center_x = -1;
	int sav_best_center_y = -1;


	/*-------------------�������к�ѡԲ�ģ��ֱ��ڹ���Բ�棨���˰ף������ڣ�ͼ�ϻ���ɫ��Բ�������Ƶ�Բ�͹���Բ�治�غϵĲ��֣�����ɫ���֣��ж��٣��Դ���ȷ����õ�Բ��--------*/

	for (unsigned int i = 0; i < probb_centers.size(); i++)
	{
		Mat fill_circle_black;
		draw_max_unidom.copyTo(fill_circle_black);//����Բ���ͼ//�����Ǻ�ɫ������Բ���ǰ�ɫ
		Point probb_centers_int((int)(probb_centers[i].x + 0.5), (int)(probb_centers[i].y + 0.5));//ȡ����ѡԲ��
		circle(fill_circle_black, probb_centers_int, radius, Scalar(0), -1);//�Ժ�ѡԲ��ΪԲ�Ļ���ɫ��Բ
		//imshow("fill_circle_black",fill_circle_black);
		//waitKey();

		//����û��Ƶ�Բ������Բ���ʣ��İ�ɫ�����ж��٣�Խ��˵�����Ƶ�ԲԽ�ӽ���ʵ�Ĺ���Բ�棩
		int frag = 0;
		for (int j = encls_rec_y_min; j < encls_rec_y_max; j++)
			for (int k = encls_rec_x_min; k < encls_rec_x_max; k++)
			{
				if (fill_circle_black.at<uchar>(j, k) == 255)
					frag++;
			}

		//����Сʣ�ಿ�ֵ�Բ������
		if (frag < min_frag)
		{
			min_frag = frag;
			sav_best_center_x = probb_centers_int.x;
			sav_best_center_y = probb_centers_int.y;
		}
	}

	/*--------�����Բ�ĵ�8���򣬻��ƺ�ɫԲȥ������Բ�棨�����ڣ�Բ��ף�����ʣ�ಿ�ֶ��٣�����һ��������-----*/
	//����ȷ����Բ�ĵ�8�����ҵ����õ�Բ��
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
	���ţ��Ŵ�뾶���ҵ����õİ뾶��
	��ʱû�а취ʵ��
	********************************/
	//��������ɫ��Բ��Ӿ���(�۳�Բ���)�ı�������̫����λʧ��
	float frag_percentage = (float)min_frag / (float)((encls_rec_x_max - encls_rec_x_min)*(encls_rec_y_max - encls_rec_y_min));
	if (frag_percentage >= 0.1)
	{
		printf("Բ�Ķ�λʧ�ܣ����ಿ�ݹ���");
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
	//�����㣬��ͼ�е�������ȥ���ƻ�
	Mat fiber_cls_element = getStructuringElement(MORPH_ELLIPSE,
		Size(2 * CLOSE_SIZE + 1, 2 * CLOSE_SIZE + 1),
		Point(CLOSE_SIZE, CLOSE_SIZE));
	morphologyEx(fiber_img_gray, fiber_img_gray, MORPH_CLOSE, fiber_cls_element, Point(-1, -1), 1);//
	Mat fiber_img_2val;
	//OTSU����Ӧ��ֵ�ָ�
	threshold(fiber_img_gray, fiber_img_2val, 0, 255, THRESH_OTSU);

	//�ҳ�������ͨ����Ϊ������Բ
	vector<vector<Point>> fiber_unidom;
	int fiber_unidom_sum = 0;
	region_grow_2(fiber_img_2val, &fiber_unidom_sum, &fiber_unidom);//��������������ͨ����ĵ����fiber_unidom
	nth_element(fiber_unidom.begin(), fiber_unidom.begin(), fiber_unidom.end(), fiberPointSizeGreater);//����ͨ��������ɴ�С��������

	//����������ͨ�򣬼����˵�Բ��
	Mat draw_max_unidom = Mat::Mat(fiber_img_2val.rows, fiber_img_2val.cols, fiber_img_2val.type(), Scalar(0, 0, 0));
	for (unsigned int i = 0; i < fiber_unidom[0].size(); i++)//ȡ�Ѿ��ź�˳�����������ͨ��ΪԲ����ȡfiber_unidom[0]
		draw_max_unidom.at<uchar>(fiber_unidom[0][i].y, fiber_unidom[0][i].x) = 255;//��draw_max_unidom�ϻ�����������ͨ���򣨹���Բ�棩

	/*-----------------��ת����ͶӰ��Ѱ�ҹ���Բ��------------------------*/
	//��һ����ķ��λ���rot_square�������˻���fiber_sqr_roi_rec�������������룬�Ա㽫����Բ�������ת��ͶӰ�������¹���Բ�澭��ת�󳬳�����
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
	vector<Point2f> probb_centers;//�����п��ܵ�Բ��
	short probb_radius;//�����п��ܵİ뾶
	find_circle_centers(rot_square, &probb_centers, &probb_radius);//��תԲ�棬����ˮƽ��ֱͶӰ�����ͶӰ�����ĵ���Ϊ����Բ��
	ad_center_point(fiber_sqr_roi_rec.width, fiber_sqr_roi_rec.height, rot_square.rows, &probb_centers);//�޳�����Բ��
	rot_square.release();

	/*--------����Բ�ĵ�λ��------------*/
	Point best_point(-1, -1);
	short best_radius = -1;
	int finder_complete = find_best_center(draw_max_unidom, probb_centers, probb_radius, &best_point, &best_radius, fiber_eva);//���Ժ�ѡԲ�ĺ�Բ�ĵ��������ص㣬��һ���ҵ����ѵ�Բ��
	if (finder_complete == -1)
		return -1;

	//����ѵ�Բ�ĺͰ뾶
	circle_center->x = best_point.x;
	circle_center->y = best_point.y;
	*circle_radius = (int)best_radius;

	return 0;
}