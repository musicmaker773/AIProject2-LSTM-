# AIProject2-LSTM-

This AI project requires us to use an LSTM to predict COVID-19 cases in the future.
There are two png files that demonstrated this project visually. One of them was a graph. It had a blue line and an orange line. The blue line was old training data and the orange line was prediction data.
The other png was the quantitative data of the orange line.

There are some issues with this project. These issues mainly stemmed on the LSTM itself. It could only predict a day ahead and needed data to adjust accordingly. So what I did was used the epochs first (and there were 5 epochs) and then predict the next day. That prediction was written in the CSV file, and the data was trained again. Then, the next prediction was found the following day based on previous inputs of data and predictions. It does this for up to 10 days. And the dates themselves were static, unless there's another CSV file that is formatted the same way.
Also, the future dates are initialized at 0 cases on the CSV file. This way of predicting is somewhat flawed and not concrete, but it does give a foundation on where the cases were gonna go somehow.
Another issue was that the predictions always changed. It is evident due to the orange line moving slightly up or slightly down or maybe balanced in between.
I was also learning python for the first time. So, the project was already hard.
