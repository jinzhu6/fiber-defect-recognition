//////////////////////////////////////////////////////////////////////////
//2012.10����ȱ�ݼ�����
//Ŀ�꣺1ģ��or����
//		2Բ��λ�á��뾶
//		3ȱ�����������
//		4�����������
//��ע�������������Ч���ȹ���fiber_scratchfinder�ļ��������߷������ƣ���������Ĳ������û���Ľ���
//////////////////////////////////////////////////////////////////////////

#include "fiberapp_syn.h"

int fiber_plotresult(Mat img, Mat fiber_zoneA_draw, Mat fiber_zoneB_draw, Mat fiber_zoneC_draw, Point circle_center, int circle_radius);

int main()
{
	int fiber_eva = 0;
	Mat fiber_img_raw = imread("..\\..\\fiber\\fiber_pp1.jpg");//����ԭʼͼƬ

	Mat fiber_img_rgb;
	Mat fiber_img_gray;//���˵ĻҶ�ͼ

	fiber_inimg_normalize(fiber_img_raw, &fiber_img_rgb, &fiber_img_gray);//��׼������ͼƬ�ķֱ��ʵı�׼��Ϊ640x480��
	fiber_img_raw.release();

	/*-------------ͼƬģ���̶��ж�---------------*/
	Mat fiber_sto_grad = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, CV_16UC1, Scalar(0));
	int sobel_result = fiber_grad_cmt(fiber_img_gray, &fiber_sto_grad, &fiber_eva);//����ͼƬ���ݶȣ�SOBEL����ÿ�����ص��ݶ�ֵ����fiber_sto_grad
	if (fiber_err_proc(fiber_eva) != 0)//��ͼƬģ����ֱ���˳�
		return -1;

	/*---------��λ���˵�Բ��-------------*/
	Point circle_center(-1, -1);//��Բ��
	int circle_radius = 0;//��Բ�뾶
	Mat fiber_img_gray_circle_locate;
	fiber_img_gray.copyTo(fiber_img_gray_circle_locate);
	fiber_circle_locate(fiber_img_gray_circle_locate, &circle_center, &circle_radius, &fiber_eva);//Բ��λ�á��뾶���
	fiber_img_gray_circle_locate.release();
	printf("Բ��λ��(%d,%d)\n", circle_center.y, circle_center.x);
	if (fiber_err_proc(fiber_eva) != 0)//���޷��ҵ�Բ����ֱ���˳�
		return -1;

	/*----------ȱ�ݼ��--------------*/
	int zone_a_impurity = 0;//��������Aȱ�ݵ�����
	int zone_b_impurity = 0;//��������Bȱ�ݵ�����
	int zone_c_impurity = 0;//��������Cȱ�ݵ�����
	Mat fiber_zoneA_draw = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, fiber_img_gray.type());//��������A��ȱ��
	Mat fiber_zoneB_draw = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, fiber_img_gray.type());//��������B��ȱ��
	Mat fiber_zoneC_draw = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, fiber_img_gray.type());//��������C��ȱ��

	fiber_impurity(fiber_img_gray, fiber_zoneA_draw, fiber_zoneB_draw, fiber_zoneC_draw, circle_center, circle_radius, &zone_a_impurity, &zone_b_impurity, &zone_c_impurity, &fiber_eva);//ȱ�ݼ�⡢ͳ��
	fiber_plotresult(fiber_img_gray, fiber_zoneA_draw, fiber_zoneB_draw, fiber_zoneC_draw, circle_center, circle_radius);//��fiber_img_gray�ϻ������ۣ�������
	/*----------���ۼ��--------------*/
	int scratch_amount = 0;//���滮������
	fiber_scratch_finder(fiber_img_gray, circle_center, circle_radius, &scratch_amount);//��⻮��

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
			//����A�����ȱ�ݣ��ף�
			if (fiber_zoneA_draw.at<uchar>(i, j) == 0)
			{
				fiber_draw.at<uchar>(i, j * 3) = 255;
				fiber_draw.at<uchar>(i, j * 3 + 1) = 255;
				fiber_draw.at<uchar>(i, j * 3 + 2) = 255;
			}
			//����B�����ȱ�ݣ��죩
			if (fiber_zoneB_draw.at<uchar>(i, j) == 0)
			{
				fiber_draw.at<uchar>(i, j * 3) = 0;
				fiber_draw.at<uchar>(i, j * 3 + 1) = 0;
				fiber_draw.at<uchar>(i, j * 3 + 2) = 255;
			}
			//����B�����ȱ�ݣ�����
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