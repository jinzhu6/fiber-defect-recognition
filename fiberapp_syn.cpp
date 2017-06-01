//////////////////////////////////////////////////////////////////////////
//2012.10光纤缺陷检测程序
//目标：1模糊or清晰
//		2圆心位置、半径
//		3缺陷数量、面积
//		4划痕数量检测
//备注：划痕数量检测效果比工程fiber_scratchfinder的检测结果差（两者方法类似，但本程序的参数设置还须改进）
//////////////////////////////////////////////////////////////////////////

#include "fiberapp_syn.h"

int fiber_plotresult(Mat img, Mat fiber_zoneA_draw, Mat fiber_zoneB_draw, Mat fiber_zoneC_draw, Point circle_center, int circle_radius);

int main()
{
	int fiber_eva = 0;
	Mat fiber_img_raw = imread("..\\..\\fiber\\fiber_pp1.jpg");//读入原始图片

	Mat fiber_img_rgb;
	Mat fiber_img_gray;//光纤的灰度图

	fiber_inimg_normalize(fiber_img_raw, &fiber_img_rgb, &fiber_img_gray);//标准化（将图片的分辨率的标准化为640x480）
	fiber_img_raw.release();

	/*-------------图片模糊程度判定---------------*/
	Mat fiber_sto_grad = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, CV_16UC1, Scalar(0));
	int sobel_result = fiber_grad_cmt(fiber_img_gray, &fiber_sto_grad, &fiber_eva);//计算图片的梯度（SOBEL），每个像素的梯度值存于fiber_sto_grad
	if (fiber_err_proc(fiber_eva) != 0)//若图片模糊则直接退出
		return -1;

	/*---------定位光纤的圆心-------------*/
	Point circle_center(-1, -1);//存圆心
	int circle_radius = 0;//存圆半径
	Mat fiber_img_gray_circle_locate;
	fiber_img_gray.copyTo(fiber_img_gray_circle_locate);
	fiber_circle_locate(fiber_img_gray_circle_locate, &circle_center, &circle_radius, &fiber_eva);//圆心位置、半径检测
	fiber_img_gray_circle_locate.release();
	printf("圆心位置(%d,%d)\n", circle_center.y, circle_center.x);
	if (fiber_err_proc(fiber_eva) != 0)//若无法找到圆心则直接退出
		return -1;

	/*----------缺陷检测--------------*/
	int zone_a_impurity = 0;//保存区域A缺陷的数量
	int zone_b_impurity = 0;//保存区域B缺陷的数量
	int zone_c_impurity = 0;//保存区域C缺陷的数量
	Mat fiber_zoneA_draw = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, fiber_img_gray.type());//保存区域A的缺陷
	Mat fiber_zoneB_draw = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, fiber_img_gray.type());//保存区域B的缺陷
	Mat fiber_zoneC_draw = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, fiber_img_gray.type());//保存区域C的缺陷

	fiber_impurity(fiber_img_gray, fiber_zoneA_draw, fiber_zoneB_draw, fiber_zoneC_draw, circle_center, circle_radius, &zone_a_impurity, &zone_b_impurity, &zone_c_impurity, &fiber_eva);//缺陷检测、统计
	fiber_plotresult(fiber_img_gray, fiber_zoneA_draw, fiber_zoneB_draw, fiber_zoneC_draw, circle_center, circle_radius);//在fiber_img_gray上画出划痕，各区域
	/*----------划痕检测--------------*/
	int scratch_amount = 0;//保存划痕数量
	fiber_scratch_finder(fiber_img_gray, circle_center, circle_radius, &scratch_amount);//检测划痕

	return 0;
}

int fiber_plotresult(Mat img, Mat fiber_zoneA_draw, Mat fiber_zoneB_draw, Mat fiber_zoneC_draw, Point circle_center, int circle_radius)
{

	Mat fiber_draw;
	cvtColor(img, fiber_draw, CV_GRAY2BGR);
	circle(fiber_draw, Point(circle_center.x, circle_center.y), circle_radius * 25 / 120, Scalar(255, 255, 255));
	circle(fiber_draw, Point(circle_center.x, circle_center.y), circle_radius * 130 / 120, Scalar(0, 255, 0));
	circle(fiber_draw, Point(circle_center.x, circle_center.y), circle_radius * 250 / 120, Scalar(255, 0, 0));

	for (int i = 0; i < fiber_zoneA_draw.rows; i++)
		for (int j = 0; j < fiber_zoneA_draw.cols; j++)
		{
			//画出A区域的缺陷（白）
			if (fiber_zoneA_draw.at<uchar>(i, j) == 0)
			{
				fiber_draw.at<uchar>(i, j * 3) = 255;
				fiber_draw.at<uchar>(i, j * 3 + 1) = 255;
				fiber_draw.at<uchar>(i, j * 3 + 2) = 255;
			}
			//画出B区域的缺陷（红）
			if (fiber_zoneB_draw.at<uchar>(i, j) == 0)
			{
				fiber_draw.at<uchar>(i, j * 3) = 0;
				fiber_draw.at<uchar>(i, j * 3 + 1) = 0;
				fiber_draw.at<uchar>(i, j * 3 + 2) = 255;
			}
			//画出B区域的缺陷（蓝）
			if (fiber_zoneC_draw.at<uchar>(i, j) == 0)
			{
				fiber_draw.at<uchar>(i, j * 3) = 255;
				fiber_draw.at<uchar>(i, j * 3 + 1) = 0;
				fiber_draw.at<uchar>(i, j * 3 + 2) = 0;
			}
		}

	imshow("fiber_draw", fiber_draw);
	waitKey(1);

	return 0;
}