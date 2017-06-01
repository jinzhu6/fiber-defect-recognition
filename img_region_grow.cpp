#include <opencv2/opencv.hpp>
#include <math.h>
using namespace std;
using namespace cv;

#define GROW_GRAY_TH 180
vector<Point> fiber_unidom;
int x_diff_t[8] = { 1,0,-1,0,-1,1,-1,1 };
int y_diff_t[8] = { 0,1,0,-1,-1,-1,1,1 };

int region_grow(Mat* fiber_img, int* fiber_def_sum)
{
	Point fiber_unidom_temp, fiber_unidom_cur;
	int nStart = 0;
	int nEnd = 0;
	int x_diff = 0;
	int y_diff = 0;
	int i;
	int j;

	for (i = 0; i < fiber_img->rows; i++)
		for (j = 0; j < fiber_img->cols; j++)
			if (fiber_img->at<uchar>(i, j) == 0)
			{
				fiber_unidom_temp.x = i;
				fiber_unidom_temp.y = j;
				fiber_unidom.push_back(fiber_unidom_temp);
				nStart = nEnd = 0;
				while (nStart <= nEnd)
				{
					fiber_unidom_cur = fiber_unidom[nStart];
					for (int k = 0; k < 8; k++)
					{

						x_diff = fiber_unidom_cur.x + x_diff_t[k];
						y_diff = fiber_unidom_cur.y + y_diff_t[k];

						if (x_diff >= 0 && x_diff < fiber_img->rows && y_diff >= 0 && y_diff < fiber_img->cols)
						{
							//printf("enter at k is %d\n",k);
							if (fiber_img->at<uchar>(x_diff, y_diff) == 0) {
								nEnd++;
								fiber_unidom_temp.x = x_diff;
								fiber_unidom_temp.y = y_diff;
								fiber_unidom.push_back(fiber_unidom_temp);
								fiber_img->at<uchar>(x_diff, y_diff) = 255;
							}
						}
					}
					nStart++;
				}
				(*fiber_def_sum)++;
				fiber_unidom.clear();
			}
	return 0;
}

//区域生长，将联通域的坐标点存入vector（坐标已经修正）
int region_grow_2(Mat fiber_img, int* fiber_def_sum, vector<vector<Point>>* fiber_unidom_all)
{
	Point fiber_unidom_temp, fiber_unidom_cur;
	int nStart = 0;
	int nEnd = 0;
	int x_diff = 0;
	int y_diff = 0;
	int i;
	int j;

	for (i = 0; i < fiber_img.rows; i++)
		for (j = 0; j < fiber_img.cols; j++)
			if (fiber_img.at<uchar>(i, j) == 0)
			{
				fiber_unidom_temp.x = j;
				fiber_unidom_temp.y = i;
				fiber_unidom.push_back(fiber_unidom_temp);
				nStart = nEnd = 0;
				while (nStart <= nEnd)
				{
					fiber_unidom_cur = fiber_unidom[nStart];
					for (int k = 0; k < 8; k++)
					{

						x_diff = fiber_unidom_cur.x + x_diff_t[k];
						y_diff = fiber_unidom_cur.y + y_diff_t[k];

						if (x_diff >= 0 && x_diff < fiber_img.cols && y_diff >= 0 && y_diff < fiber_img.rows)
						{
							if (fiber_img.at<uchar>(y_diff, x_diff) == 0) {
								nEnd++;
								fiber_unidom_temp.x = x_diff;
								fiber_unidom_temp.y = y_diff;
								fiber_unidom.push_back(fiber_unidom_temp);
								fiber_img.at<uchar>(y_diff, x_diff) = 255;
							}
						}
					}
					nStart++;


				}
				(*fiber_def_sum)++;
				(*fiber_unidom_all).push_back(fiber_unidom);
				fiber_unidom.clear();
			}
	return 0;
}

//基于一定灰度范围的区域生长
int region_grow_basegray(Mat src, Mat* dst)
{
	vector<Point> sto_homo_zone_pt;
	vector<vector<Point>> sto_homo_zone_ass;
	Point cur_point;

	Mat mark_map = Mat::Mat(src.rows, src.cols, src.type());
	mark_map.setTo(Scalar(255));

	int zone_gray_level;
	for (int i = 1; i < src.rows - 1; i++)
		for (int j = 1; j < src.cols - 1; j++)
		{
			if (mark_map.at<uchar>(i, j) == 255 && src.at<uchar>(i, j) > GROW_GRAY_TH)
			{

				zone_gray_level = src.at<uchar>(i, j);
				mark_map.at<uchar>(i, j) = 0;
				sto_homo_zone_pt.push_back(Point(j, i));
				while (1)
				{
					if (!sto_homo_zone_pt.empty()) {
						cur_point.x = (sto_homo_zone_pt.end() - 1)->x;
						cur_point.y = (sto_homo_zone_pt.end() - 1)->y;
						sto_homo_zone_pt.pop_back();
					}
					else
						break;

					for (int k = 0; k < 8; k++)
					{
						if (mark_map.at<uchar>(cur_point.y + y_diff_t[k], cur_point.x + x_diff_t[k]) == 255)
						{
							int diff_gray_level = src.at<uchar>(cur_point.y + y_diff_t[k], cur_point.x + x_diff_t[k]);
							if (abs(diff_gray_level - zone_gray_level) <= 20)
							{
								if ((cur_point.x + x_diff_t[k]) > 1 && (cur_point.x + x_diff_t[k]) < (src.cols - 1) && (cur_point.y + y_diff_t[k]) > 1 && (cur_point.y + y_diff_t[k]) < (src.rows - 1))
									sto_homo_zone_pt.push_back(Point(cur_point.x + x_diff_t[k], cur_point.y + y_diff_t[k]));
								mark_map.at<uchar>(cur_point.y + y_diff_t[k], cur_point.x + x_diff_t[k]) = 0;
							}
						}
					}
				}
			}
		}

	mark_map.copyTo(*dst);
	mark_map.release();
	return 0;
}