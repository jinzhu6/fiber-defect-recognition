#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int fastThin(Mat src_mat, Mat* dst, int iterations)//将IPL_DEPTH_8U型二值图像进行细化
{
	Mat dst_mat = *dst;

	for (int i = 0; i < src_mat.rows; i++)
		for (int j = 0; j < src_mat.cols; j++)
		{
			if (src_mat.at<uchar>(i, j) == 255)
				src_mat.at<uchar>(i, j) = 0;
			else src_mat.at<uchar>(i, j) = 1;
		}

	src_mat.copyTo(dst_mat);

	int n = 0, i = 0, j = 0;
	for (n = 0; n < iterations; n++)
	{
		Mat t_mat = dst_mat.clone();

		for (i = 0; i < src_mat.rows; i++)
		{
			for (j = 0; j < src_mat.cols; j++)
			{
				if (t_mat.at<uchar>(i, j) == 1)
				{
					int ap = 0;
					int p2 = (i == 0) ? 0 : t_mat.at<uchar>(i - 1, j);
					int p3 = (i == 0 || j == src_mat.cols - 1) ? 0 : t_mat.at<uchar>(i - 1, j + 1);
					if (p2 == 0 && p3 == 1)
					{
						ap++;
					}
					int p4 = (j == src_mat.cols - 1) ? 0 : t_mat.at<uchar>(i, j + 1);
					if (p3 == 0 && p4 == 1)
					{
						ap++;
					}
					int p5 = (i == src_mat.rows - 1 || j == src_mat.cols - 1) ? 0 : t_mat.at<uchar>(i + 1, j + 1);
					if (p4 == 0 && p5 == 1)
					{
						ap++;
					}
					int p6 = (i == src_mat.rows - 1) ? 0 : t_mat.at<uchar>(i + 1, j);
					if (p5 == 0 && p6 == 1)
					{
						ap++;
					}
					int p7 = (i == src_mat.rows - 1 || j == 0) ? 0 : t_mat.at<uchar>(i + 1, j - 1);
					if (p6 == 0 && p7 == 1)
					{
						ap++;
					}
					int p8 = (j == 0) ? 0 : t_mat.at<uchar>(i, j - 1);
					if (p7 == 0 && p8 == 1)
					{
						ap++;
					}
					int p9 = (i == 0 || j == 0) ? 0 : t_mat.at<uchar>(i - 1, j - 1);
					if (p8 == 0 && p9 == 1)
					{
						ap++;
					}
					if (p9 == 0 && p2 == 1)
					{
						ap++;
					}
					if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) > 1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) < 7)
					{
						if (ap == 1)
						{
							if (p2 * p4 * p6 == 0)
							{
								if (p4 * p6 * p8 == 0)
								{
									dst_mat.at<uchar>(i, j) = 0;
								}
							}
						}
					}
				}
			}
		}

		t_mat.release();
		t_mat = dst_mat.clone();

		for (i = 0; i < src_mat.rows; i++)
		{
			for (int j = 0; j < src_mat.cols; j++)
			{
				if (t_mat.at<uchar>(i, j) == 1)
				{
					int ap = 0;
					int p2 = (i == 0) ? 0 : t_mat.at<uchar>(i - 1, j);
					int p3 = (i == 0 || j == src_mat.cols - 1) ? 0 : t_mat.at<uchar>(i - 1, j + 1);
					if (p2 == 0 && p3 == 1)
					{
						ap++;
					}
					int p4 = (j == src_mat.cols - 1) ? 0 : t_mat.at<uchar>(i, j + 1);
					if (p3 == 0 && p4 == 1)
					{
						ap++;
					}
					int p5 = (i == src_mat.rows - 1 || j == src_mat.cols - 1) ? 0 : t_mat.at<uchar>(i + 1, j + 1);
					if (p4 == 0 && p5 == 1)
					{
						ap++;
					}
					int p6 = (i == src_mat.rows - 1) ? 0 : t_mat.at<uchar>(i + 1, j);
					if (p5 == 0 && p6 == 1)
					{
						ap++;
					}
					int p7 = (i == src_mat.rows - 1 || j == 0) ? 0 : t_mat.at<uchar>(i + 1, j - 1);
					if (p6 == 0 && p7 == 1)
					{
						ap++;
					}
					int p8 = (j == 0) ? 0 : t_mat.at<uchar>(i, j - 1);
					if (p7 == 0 && p8 == 1)
					{
						ap++;
					}
					int p9 = (i == 0 || j == 0) ? 0 : t_mat.at<uchar>(i - 1, j - 1);
					if (p8 == 0 && p9 == 1)
					{
						ap++;
					}
					if (p9 == 0 && p2 == 1)
					{
						ap++;
					}
					if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) > 1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) < 7)
					{
						if (ap == 1)
						{
							if (p2*p4*p8 == 0)
							{
								if (p2*p6*p8 == 0)
								{
									dst_mat.at<uchar>(i, j) = 0;
								}
							}
						}
					}
				}

			}

		}
		t_mat.release();
	}

	for (int i = 0; i < dst_mat.rows; i++)
		for (int j = 0; j < dst_mat.cols; j++)
		{
			if (dst_mat.at<uchar>(i, j) == 1)
				dst_mat.at<uchar>(i, j) = 255;
			else dst_mat.at<uchar>(i, j) = 0;
		}

	return 0;
}
