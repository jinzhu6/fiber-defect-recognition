#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
#include "img_region_grow.h"
#include "img_io_trans.h"

#define  MORPH_SIZE 5
#define  ZONE_A_DIA    25.0
#define  ZONE_B_DIA	   120.0
#define  ZONE_BC_DIA   130.0
#define  ZONE_C_DIA	   250.0

#define  HIS_LOW_T     100
#define  DEF_THRESH_PRO 0.714
#define  MORPH_SIZE 5

int fiber_zone_cmt_pixsum(Mat fiber_zoneA)
{
	int pix_sum = 0;
	for (int i = 0; i < fiber_zoneA.rows; i++)
		for (int j = 0; j < fiber_zoneA.cols; j++) {
			if (fiber_zoneA.at<uchar>(i, j) == 0)
				pix_sum++;
		}
	return pix_sum;
}

int histogram_cmt(Mat fiber_img_cc_gray_merge, int histogram[256])
{
	for (int i = 0; i < fiber_img_cc_gray_merge.rows; i++)
		for (int j = 0; j < fiber_img_cc_gray_merge.cols; j++) {
			if (fiber_img_cc_gray_merge.at<uchar>(i, j) != 255)
				histogram[fiber_img_cc_gray_merge.at<uchar>(i, j)]++;
		}

	//平滑坐标,从左往右从右往左滤波
	for (int i = 1; i < 256; i++)
		histogram[i] = int(histogram[i - 1] * 0.7 + histogram[i] * 0.3 + 0.5);
	for (int i = 254; i >= 0; i--)
		histogram[i] = int(histogram[i + 1] * 0.7 + histogram[i] * 0.3 + 0.5);

	//画出直方图
	Mat histogram_show = Mat::Mat(256, 1000, fiber_img_cc_gray_merge.type(), Scalar(0, 0, 0));
	for (int i = 0; i < 256; i++) {
		Point line_op;
		line_op.x = 0;
		line_op.y = i;
		Point line_ed;
		line_ed.x = histogram[i];
		line_ed.y = i;
		line(histogram_show, line_op, line_ed, Scalar(255, 255, 255));
	}

	return 0;
}

int find_zoneB_th(int histogram[256], int* zone_AB_defect_thresh)
{
	int   fiber_zoneB_max_peak = 0;
	int fiber_zoneB_max_peak_val = 0;
	for (int i = 0; i < 256; i++)
	{
		if (fiber_zoneB_max_peak_val < histogram[i])
		{
			fiber_zoneB_max_peak_val = histogram[i];
			fiber_zoneB_max_peak = i;
		}
	}
	for (int i = fiber_zoneB_max_peak; i > 3; i--)
	{
		if (histogram[i] < HIS_LOW_T
			&& histogram[i - 1] < HIS_LOW_T
			&& histogram[i - 2] < HIS_LOW_T
			&& histogram[i - 3] < HIS_LOW_T
			&& histogram[i - 4] < HIS_LOW_T
			)//100是自己设置的一个阈值
		{
			*zone_AB_defect_thresh = i;
			break;
		}
	}
	(*zone_AB_defect_thresh) = (int)((float)(*zone_AB_defect_thresh)*DEF_THRESH_PRO + 0.5);
	return 0;
}

int ring_gray_level_cmt(Mat fiber_img_gray, Mat fiber_zbc_mask_inv, Mat fiber_zb_mask, int* ring_gray_level)
{
	Mat fiber_morph_element = getStructuringElement(MORPH_ELLIPSE, //开运算所用元素
		Size(2 * MORPH_SIZE + 1, 2 * MORPH_SIZE + 1),
		Point(MORPH_SIZE, MORPH_SIZE));
	dilate(fiber_img_gray, fiber_img_gray, fiber_morph_element, Point(-1, -1), 1);
	erode(fiber_img_gray, fiber_img_gray, fiber_morph_element, Point(-1, -1), 1);
	fiber_img_gray.setTo(Scalar(0), fiber_zbc_mask_inv);
	fiber_img_gray.setTo(Scalar(0), fiber_zb_mask);

	int ring_gray_val = 0;
	int ring_pixel = 0;
	for (int i = 0; i < fiber_img_gray.rows; i++)
		for (int j = 0; j < fiber_img_gray.cols; j++)
		{
			if (fiber_img_gray.at<uchar>(i, j) != 0)
			{
				ring_gray_val += fiber_img_gray.at<uchar>(i, j);
				ring_pixel++;
			}
		}

	ring_gray_val = int((float)ring_gray_val / (float)ring_pixel + 0.5);
	*ring_gray_level = ring_gray_val;

	return  0;
}
int fiber_impurity(Mat fiber_img_gray, Mat fiber_zoneA_draw, Mat fiber_zoneB_draw, Mat fiber_zoneC_draw, Point circle_center, int circle_radius, int* zone_a_impurity, int* zone_b_impurity, int* zone_c_impurity, int* fiber_eva)
{
	/*--------该函数的思路是准备好光纤各个区域的mask图，利用MASK图将光纤每个区域抠出来，用二值分割得到缺陷-----------*/
	Mat fiber_img_gray_bk;
	fiber_img_gray.copyTo(fiber_img_gray_bk);
	img_enhance(&fiber_img_gray);//底顶帽变换，增强图片对比度

	//计算各区域的半径长
	int fiber_zbc_r = (int)(circle_radius / ZONE_B_DIA*ZONE_BC_DIA);
	int fiber_zb_r = circle_radius;
	int fiber_zc_r = (int)(circle_radius / ZONE_B_DIA*ZONE_C_DIA);
	int fiber_za_r = (int)(circle_radius / ZONE_B_DIA*ZONE_A_DIA);

	//准备所有区域的mask图
	Mat fiber_za_mask = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, fiber_img_gray.type(), Scalar(0));
	Mat fiber_zb_mask_pp = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, fiber_img_gray.type(), Scalar(0));
	Mat fiber_zb_mask = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, fiber_img_gray.type(), Scalar(0));
	Mat fiber_zc_mask = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, fiber_img_gray.type(), Scalar(0));
	Mat fiber_za_mask_inv = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, fiber_img_gray.type(), Scalar(0));
	Mat fiber_zb_mask_inv = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, fiber_img_gray.type(), Scalar(0));
	Mat fiber_zb_mask_inv_pp = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, fiber_img_gray.type(), Scalar(0));
	Mat fiber_zc_mask_inv = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, fiber_img_gray.type(), Scalar(0));
	Mat fiber_zbc_mask = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, fiber_img_gray.type(), Scalar(0));
	Mat fiber_zbc_mask_inv = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, fiber_img_gray.type(), Scalar(0));
	circle(fiber_za_mask, circle_center, fiber_za_r, Scalar(255), -1);//背景黑前景白//参数-1（thinness）是画实心的圆//前景白背景黑
	circle(fiber_zb_mask, circle_center, fiber_zb_r, Scalar(255), -1);//背景黑前景白
	circle(fiber_zb_mask_pp, circle_center, fiber_zb_r - 10, Scalar(255), -1);//背景黑前景白
	circle(fiber_zc_mask, circle_center, fiber_zc_r, Scalar(255), -1);//背景黑前景白
	circle(fiber_zbc_mask, circle_center, fiber_zbc_r, Scalar(255), -1);//背景黑前景白
	bitwise_not(fiber_za_mask, fiber_za_mask_inv);//背景白前景黑
	bitwise_not(fiber_zb_mask, fiber_zb_mask_inv);//背景白前景黑
	bitwise_not(fiber_zb_mask_pp, fiber_zb_mask_inv_pp);
	bitwise_not(fiber_zc_mask, fiber_zc_mask_inv);//背景白前景黑
	bitwise_not(fiber_zbc_mask, fiber_zbc_mask_inv);

	int fiber_def_a_sum = 0;//存A区域缺陷数量
	int fiber_def_b_sum = 0;//存B区域缺陷数量
	int fiber_def_c_sum = 0;//存C区域缺陷数量
	Mat fiber_img_cc_gray_merge = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, fiber_img_gray.type(), Scalar(0));
	Mat fiber_def_a_copy;//备份区域A的缺陷图
	Mat fiber_def_b_copy;//备份区域B的缺陷图
	Mat fiber_def_c_copy;//备份区域C的缺陷图

	/*---------先利用B区域计算出分割的域值，用于A,B区域的分割，C区域将使用自适应域值------------*/
	fiber_img_gray.copyTo(fiber_img_cc_gray_merge);
	fiber_img_cc_gray_merge.setTo(255, fiber_zb_mask_inv);

	//计算除255灰度外的所有像素的灰度直方图
	int histogram[256] = { 0 };
	histogram_cmt(fiber_img_cc_gray_merge, histogram);

	int zone_AB_defect_thresh = 0;
	//利用直方图(双峰法)找到适用于A和B区域分割的阈值
	find_zoneB_th(histogram, &zone_AB_defect_thresh);

	/*-------------------------------A区域的处理--------------------*/
	fiber_img_gray.copyTo(fiber_img_cc_gray_merge);
	fiber_img_cc_gray_merge.setTo(255, fiber_za_mask_inv);//用MASK图fiber_za_mask_inv抠出区域A的图像
	threshold(fiber_img_cc_gray_merge, fiber_img_cc_gray_merge, zone_AB_defect_thresh, 255, THRESH_BINARY);//对抠出的图实施二值化，用双峰法找到的域值
	fiber_img_cc_gray_merge.copyTo(fiber_def_a_copy);//备份A区域的缺陷
	fiber_img_cc_gray_merge.copyTo(fiber_zoneA_draw);//输出缺陷

	int zoneA_def_pixsum = fiber_zone_cmt_pixsum(fiber_zoneA_draw);//计算A区域缺陷的像素个数
	int zoneA_pixsum = fiber_zone_cmt_pixsum(fiber_za_mask_inv);//计算A区面积
	printf("A区缺陷占A区面积%f%%\n", (float)zoneA_def_pixsum / (float)zoneA_pixsum*100.0);

	region_grow(&fiber_img_cc_gray_merge, &fiber_def_a_sum);//输出A区域的联通域（缺陷）个数
	printf("区域A内检测到缺陷%d个\n", fiber_def_a_sum);

	/*-------------------------B区域的处理-----------------------*/
	fiber_img_gray.copyTo(fiber_img_cc_gray_merge);
	fiber_img_cc_gray_merge.setTo(255, fiber_za_mask);
	fiber_img_cc_gray_merge.setTo(255, fiber_zb_mask_inv_pp);//用MASK图fiber_za_mask_inv抠出区域B的图像
	threshold(fiber_img_cc_gray_merge, fiber_img_cc_gray_merge, zone_AB_defect_thresh, 255, THRESH_BINARY);

	fiber_img_cc_gray_merge.copyTo(fiber_def_b_copy);//备份B区域的缺陷
	fiber_img_cc_gray_merge.copyTo(fiber_zoneB_draw);//输出缺陷

	int zoneB_def_pixsum = fiber_zone_cmt_pixsum(fiber_zoneB_draw);//计算B区域缺陷的像素个数
	int zoneB_pixsum = fiber_zone_cmt_pixsum(fiber_zb_mask_inv);//计算B区面积
	zoneB_def_pixsum += zoneA_def_pixsum;
	printf("B区内缺陷（含A区缺陷）占B区面积%f%%\n", (float)zoneB_def_pixsum / (float)zoneB_pixsum*100.0);
	region_grow(&fiber_img_cc_gray_merge, &fiber_def_b_sum);//区域生长，输出联通域个数
	printf("区域B内检测到缺陷%d个\n", fiber_def_b_sum);

	/*-----------------c区域处理，用原图，利用自适应阈值分割---------------------*/
	int ring_gray_level = 0;
	fiber_img_gray_bk.copyTo(fiber_img_cc_gray_merge);

	ring_gray_level_cmt(fiber_img_gray_bk, fiber_zbc_mask_inv, fiber_zb_mask, &ring_gray_level);//计算环形区域的灰度均值//环形区域是bc区域扣去b区域剩下的区域
	//fiber_zb_r+1是防之前的圆半径太小从而出现黑色边缘
	circle(fiber_img_cc_gray_merge, circle_center, fiber_zb_r + 1, Scalar(ring_gray_level), -1);//用上一步计算好的灰度值填充圆面，之所以用这个灰度值填充圆面是为了防止原来的圆面灰度太低而导致圆的边缘部分被自适应分割错判定为缺陷//背景黑前景白

	adaptiveThreshold(fiber_img_cc_gray_merge, fiber_img_cc_gray_merge, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 71, 27);//分块自适应阈值分割（有益于克服光照）

	fiber_img_cc_gray_merge.setTo(255, fiber_zbc_mask);//消除bc区域外的所有图像
	fiber_img_cc_gray_merge.setTo(255, fiber_zc_mask_inv);
	fiber_img_cc_gray_merge.copyTo(fiber_def_c_copy);//备份C区域的缺陷
	fiber_img_cc_gray_merge.copyTo(fiber_zoneC_draw);//输出缺陷
	region_grow(&fiber_img_cc_gray_merge, &fiber_def_c_sum);//区域生长，输出c区域联通区域（缺陷）的个数
	printf("区域C内检测到缺陷%d个\n", fiber_def_c_sum);

	/*-----存缺陷的数量---*/
	*zone_a_impurity = fiber_def_a_sum;
	*zone_b_impurity = fiber_def_b_sum;
	*zone_c_impurity = fiber_def_c_sum;

	return 0;
}
