var myChart = window.echarts.init(document.getElementById('mnist_test'));

var dataSet = {};
d3.csv("./res-mnist.csv", function(data) {
    data.forEach(function(d, i) {
        d.id = Math.round(Number(+d.id));
        d.dim1 = +d.dim1;
        d.dim2 = +d.dim2;
        d.label = parseInt(d.label_pred);
        d.p_y = +d.p_y;
        d.p_y_ = [+d.p_y_0, +d.p_y_1, +d.p_y_2, +d.p_y_3, +d.p_y_4,
            +d.p_y_5, +d.p_y_6, +d.p_y_7, +d.p_y_8, +d.p_y_9];
        if (!dataSet[d.label]) {
            dataSet[d.label] = [];
        }
        dataSet[d.label].push(d);
    });
    var labels = Object.keys(dataSet);
    var series = [];
    labels.forEach(function (l) {
        var temp = {};
        temp['Name'] = l;
        temp['type'] = 'scatter';
        temp['symbolSize'] = 5;
        temp['data'] = dataSet[l].map(function (d) {
            return [d.dim1, d.dim2];
        });
        temp['animation'] = false;
        series.push(temp);
    });
    var option = {
        grid: {
            left: '3%',
            right: '7%',
            bottom: '3%',
            containLabel: true
        },
        toolbox: {
            feature: {
                dataZoom: {},
                brush: {
                    type: ['rect', 'polygon', 'clear']
                }
            }
        },
        brush: {
        },
        legend: {
            data: labels,
            left: 'center'
        },
        xAxis : {show: false},
        yAxis : {show: false},
        series : series
    };

    myChart.setOption(option);

    myChart.on('click', function (params) {
        console.log(params);
    });

});
