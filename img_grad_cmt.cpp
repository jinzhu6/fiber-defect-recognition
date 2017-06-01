#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define SOBEL_TH_X 250
#define SOBEL_TH_Y SOBEL_TH_X*640/480
#define BORDER 14
#define IMG_DSITN_TH 50

int fiber_grad_cmt(Mat fiber_img_gray, Mat* fiber_sto_grad, int* fiber_eva)
{
	int sum_sobel = 0;

	for (int i = BORDER + 1; i < fiber_img_gray.rows - BORDER - 1; i++)
		for (int j = BORDER + 1; j < fiber_img_gray.cols - BORDER - 1; j++)

		{
			int sto_sobel = 0;
			int sobel_x = 0, sobel_y = 0;
			/*-------计算每一个像素的SOBEL梯度-----*/
			sobel_x = abs(fiber_img_gray.at<uchar>(i - 1, j + 1)
				+ 2 * fiber_img_gray.at<uchar>(i, j + 1)
				+ fiber_img_gray.at<uchar>(i + 1, j + 1)
				- fiber_img_gray.at<uchar>(i - 1, j - 1)
				- 2 * fiber_img_gray.at<uchar>(i, j - 1)
				- fiber_img_gray.at<uchar>(i + 1, j - 1));

			sobel_y = abs(fiber_img_gray.at<uchar>(i - 1, j - 1)
				+ 2 * fiber_img_gray.at<uchar>(i - 1, j)
				+ fiber_img_gray.at<uchar>(i - 1, j + 1)
				- fiber_img_gray.at<uchar>(i + 1, j - 1)
				- 2 * fiber_img_gray.at<uchar>(i + 1, j)
				- fiber_img_gray.at<uchar>(i + 1, j + 1));

			sto_sobel = sobel_x + sobel_y;
			(*fiber_sto_grad).at<short>(i, j) = sto_sobel;

			/*------抑制较小梯度值-----*/
			if (sobel_x < SOBEL_TH_X)
				sobel_x = 0;
			if (sobel_y < SOBEL_TH_Y)
				sobel_y = 0;
			sum_sobel += sobel_x + sobel_y;

		}


	if (sum_sobel / 100.0 > IMG_DSITN_TH)
		;
	else
	{
		(*fiber_eva) |= 0x001;//图片模糊，置fiber_eva的标志位
	}

	return sum_sobel;
}