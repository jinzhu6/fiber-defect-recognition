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

	//ƽ������,�������Ҵ��������˲�
	for (int i = 1; i < 256; i++)
		histogram[i] = int(histogram[i - 1] * 0.7 + histogram[i] * 0.3 + 0.5);
	for (int i = 254; i >= 0; i--)
		histogram[i] = int(histogram[i + 1] * 0.7 + histogram[i] * 0.3 + 0.5);

	//����ֱ��ͼ
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
			)//100���Լ����õ�һ����ֵ
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
	Mat fiber_morph_element = getStructuringElement(MORPH_ELLIPSE, //����������Ԫ��
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
	/*--------�ú�����˼·��׼���ù��˸��������maskͼ������MASKͼ������ÿ������ٳ������ö�ֵ�ָ�õ�ȱ��-----------*/
	Mat fiber_img_gray_bk;
	fiber_img_gray.copyTo(fiber_img_gray_bk);
	img_enhance(&fiber_img_gray);//�׶�ñ�任����ǿͼƬ�Աȶ�

	//���������İ뾶��
	int fiber_zbc_r = (int)(circle_radius / ZONE_B_DIA*ZONE_BC_DIA);
	int fiber_zb_r = circle_radius;
	int fiber_zc_r = (int)(circle_radius / ZONE_B_DIA*ZONE_C_DIA);
	int fiber_za_r = (int)(circle_radius / ZONE_B_DIA*ZONE_A_DIA);

	//׼�����������maskͼ
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
	circle(fiber_za_mask, circle_center, fiber_za_r, Scalar(255), -1);//������ǰ����//����-1��thinness���ǻ�ʵ�ĵ�Բ//ǰ���ױ�����
	circle(fiber_zb_mask, circle_center, fiber_zb_r, Scalar(255), -1);//������ǰ����
	circle(fiber_zb_mask_pp, circle_center, fiber_zb_r - 10, Scalar(255), -1);//������ǰ����
	circle(fiber_zc_mask, circle_center, fiber_zc_r, Scalar(255), -1);//������ǰ����
	circle(fiber_zbc_mask, circle_center, fiber_zbc_r, Scalar(255), -1);//������ǰ����
	bitwise_not(fiber_za_mask, fiber_za_mask_inv);//������ǰ����
	bitwise_not(fiber_zb_mask, fiber_zb_mask_inv);//������ǰ����
	bitwise_not(fiber_zb_mask_pp, fiber_zb_mask_inv_pp);
	bitwise_not(fiber_zc_mask, fiber_zc_mask_inv);//������ǰ����
	bitwise_not(fiber_zbc_mask, fiber_zbc_mask_inv);

	int fiber_def_a_sum = 0;//��A����ȱ������
	int fiber_def_b_sum = 0;//��B����ȱ������
	int fiber_def_c_sum = 0;//��C����ȱ������
	Mat fiber_img_cc_gray_merge = Mat::Mat(fiber_img_gray.rows, fiber_img_gray.cols, fiber_img_gray.type(), Scalar(0));
	Mat fiber_def_a_copy;//��������A��ȱ��ͼ
	Mat fiber_def_b_copy;//��������B��ȱ��ͼ
	Mat fiber_def_c_copy;//��������C��ȱ��ͼ

	/*---------������B���������ָ����ֵ������A,B����ķָC����ʹ������Ӧ��ֵ------------*/
	fiber_img_gray.copyTo(fiber_img_cc_gray_merge);
	fiber_img_cc_gray_merge.setTo(255, fiber_zb_mask_inv);

	//�����255�Ҷ�����������صĻҶ�ֱ��ͼ
	int histogram[256] = { 0 };
	histogram_cmt(fiber_img_cc_gray_merge, histogram);

	int zone_AB_defect_thresh = 0;
	//����ֱ��ͼ(˫�巨)�ҵ�������A��B����ָ����ֵ
	find_zoneB_th(histogram, &zone_AB_defect_thresh);

	/*-------------------------------A����Ĵ���--------------------*/
	fiber_img_gray.copyTo(fiber_img_cc_gray_merge);
	fiber_img_cc_gray_merge.setTo(255, fiber_za_mask_inv);//��MASKͼfiber_za_mask_inv�ٳ�����A��ͼ��
	threshold(fiber_img_cc_gray_merge, fiber_img_cc_gray_merge, zone_AB_defect_thresh, 255, THRESH_BINARY);//�Կٳ���ͼʵʩ��ֵ������˫�巨�ҵ�����ֵ
	fiber_img_cc_gray_merge.copyTo(fiber_def_a_copy);//����A�����ȱ��
	fiber_img_cc_gray_merge.copyTo(fiber_zoneA_draw);//���ȱ��

	int zoneA_def_pixsum = fiber_zone_cmt_pixsum(fiber_zoneA_draw);//����A����ȱ�ݵ����ظ���
	int zoneA_pixsum = fiber_zone_cmt_pixsum(fiber_za_mask_inv);//����A�����
	printf("A��ȱ��ռA�����%f%%\n", (float)zoneA_def_pixsum / (float)zoneA_pixsum*100.0);

	region_grow(&fiber_img_cc_gray_merge, &fiber_def_a_sum);//���A�������ͨ��ȱ�ݣ�����
	printf("����A�ڼ�⵽ȱ��%d��\n", fiber_def_a_sum);

	/*-------------------------B����Ĵ���-----------------------*/
	fiber_img_gray.copyTo(fiber_img_cc_gray_merge);
	fiber_img_cc_gray_merge.setTo(255, fiber_za_mask);
	fiber_img_cc_gray_merge.setTo(255, fiber_zb_mask_inv_pp);//��MASKͼfiber_za_mask_inv�ٳ�����B��ͼ��
	threshold(fiber_img_cc_gray_merge, fiber_img_cc_gray_merge, zone_AB_defect_thresh, 255, THRESH_BINARY);

	fiber_img_cc_gray_merge.copyTo(fiber_def_b_copy);//����B�����ȱ��
	fiber_img_cc_gray_merge.copyTo(fiber_zoneB_draw);//���ȱ��

	int zoneB_def_pixsum = fiber_zone_cmt_pixsum(fiber_zoneB_draw);//����B����ȱ�ݵ����ظ���
	int zoneB_pixsum = fiber_zone_cmt_pixsum(fiber_zb_mask_inv);//����B�����
	zoneB_def_pixsum += zoneA_def_pixsum;
	printf("B����ȱ�ݣ���A��ȱ�ݣ�ռB�����%f%%\n", (float)zoneB_def_pixsum / (float)zoneB_pixsum*100.0);
	region_grow(&fiber_img_cc_gray_merge, &fiber_def_b_sum);//���������������ͨ�����
	printf("����B�ڼ�⵽ȱ��%d��\n", fiber_def_b_sum);

	/*-----------------c��������ԭͼ����������Ӧ��ֵ�ָ�---------------------*/
	int ring_gray_level = 0;
	fiber_img_gray_bk.copyTo(fiber_img_cc_gray_merge);

	ring_gray_level_cmt(fiber_img_gray_bk, fiber_zbc_mask_inv, fiber_zb_mask, &ring_gray_level);//���㻷������ĻҶȾ�ֵ//����������bc�����ȥb����ʣ�µ�����
	//fiber_zb_r+1�Ƿ�֮ǰ��Բ�뾶̫С�Ӷ����ֺ�ɫ��Ե
	circle(fiber_img_cc_gray_merge, circle_center, fiber_zb_r + 1, Scalar(ring_gray_level), -1);//����һ������õĻҶ�ֵ���Բ�棬֮����������Ҷ�ֵ���Բ����Ϊ�˷�ֹԭ����Բ��Ҷ�̫�Ͷ�����Բ�ı�Ե���ֱ�����Ӧ�ָ���ж�Ϊȱ��//������ǰ����

	adaptiveThreshold(fiber_img_cc_gray_merge, fiber_img_cc_gray_merge, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 71, 27);//�ֿ�����Ӧ��ֵ�ָ�����ڿ˷����գ�

	fiber_img_cc_gray_merge.setTo(255, fiber_zbc_mask);//����bc�����������ͼ��
	fiber_img_cc_gray_merge.setTo(255, fiber_zc_mask_inv);
	fiber_img_cc_gray_merge.copyTo(fiber_def_c_copy);//����C�����ȱ��
	fiber_img_cc_gray_merge.copyTo(fiber_zoneC_draw);//���ȱ��
	region_grow(&fiber_img_cc_gray_merge, &fiber_def_c_sum);//�������������c������ͨ����ȱ�ݣ��ĸ���
	printf("����C�ڼ�⵽ȱ��%d��\n", fiber_def_c_sum);

	/*-----��ȱ�ݵ�����---*/
	*zone_a_impurity = fiber_def_a_sum;
	*zone_b_impurity = fiber_def_b_sum;
	*zone_c_impurity = fiber_def_c_sum;

	return 0;
}
