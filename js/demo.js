var myChart = window.echarts.init(document.getElementById('demo_chart'));
var dataSet = {};
var orgininal = [];

d3.csv(dataCSV, function (data) {
    data.forEach(function (d, i) {
        d.id = Math.round(Number(+d.id));
        d.dim1 = +d.dim1;
        d.dim2 = +d.dim2;
        d.index = i;
        d.label = parseInt(d.label_pred);
        d.p_y = +d.p_y;
        d.p_y_ = [+d.p_y_0, +d.p_y_1, +d.p_y_2, +d.p_y_3, +d.p_y_4,
        +d.p_y_5, +d.p_y_6, +d.p_y_7, +d.p_y_8, +d.p_y_9];
        if (!dataSet[d.label]) {
            dataSet[d.label] = [];
        }
        orgininal.push(d);
        dataSet[d.label].push(d);
    });
    var labels = Object.keys(dataSet);
    var series = [];
    var legends = labels.map(function (el) {
        return {
            name: labels[el]
        }
    });

    labels.forEach(function (l, i) {
        var temp = {};
        temp['name'] = labels[parseInt(l)];
        temp['type'] = 'scatter';
        temp['symbolSize'] = 5;
        temp['data'] = dataSet[l].map(function (d) {
            return [d.dim1, d.dim2, d.index];
        });
        // temp['animation'] = false;
        series.push(temp);
    });
    var option = {
        grid: {
            left: '0',
            right: '5%',
            bottom: '5%',
            containLabel: true
        },
        dataZoom: [
            {
                type: 'slider',
                show: true,
                xAxisIndex: [0]
            },
            {
                type: 'slider',
                show: true,
                yAxisIndex: [0]
            },
            {
                type: 'inside',
                show: true,
                xAxisIndex: [0]
            },
            {
                type: 'inside',
                show: true,
                yAxisIndex: [0]
            }
        ],
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
            data: legends
        },
        xAxis: { show: false },
        yAxis: { show: false },
        series: series
    };
    myChart.setOption(option);
    var colorMap = myChart.getOption().color;

    colorMap.forEach(function (el, i) {
        var _id = '#chart-label-' + i;
        $(_id).css('background-color', el);
    });

    myChart.on('click', function (params) {
        var dataIndex = params['data'][2];
        var dataPoint = orgininal[dataIndex];
        console.log(params, dataPoint);
        $("#img-container").attr('src', imgFolder + dataPoint.id + '.jpg');
        dataPoint.p_y_.forEach(function (c, i) {
            var label_indicator_id = "#chart-label-" + i;
            var _label_id = "#label-" + i;
            var label_value_id = "#value-" + i;
            var new_width = 120 * c + 'px';
            $(label_indicator_id).animate({ 'width': new_width });
            var f = parseFloat(c);
            $(label_value_id).text(+(f.toFixed(6)));
            $(_label_id).css('font-size', '12px');
            $(label_id).css('color', '#000');
        });
        var label_id = "#label-" + dataPoint.label;
        $(label_id).animate({ 'font-size': '18px' });
        $("#py").text(dataPoint.p_y);
    });

});
