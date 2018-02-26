var margin = { top: 20, right: 20, bottom: 30, left: 40 },
    width = 1024 - margin.left - margin.right,
    widthPlot = width - 256,
    height = 512 - margin.top - margin.bottom;

var color = d3.scale.category10();

var x = d3.scale.linear()
    .range([0, widthPlot]);

var y = d3.scale.linear()
    .range([height, 0]);

var xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom");

var yAxis = d3.svg.axis()
    .scale(y)
    .orient("left");

var svg = d3.select("#mnist_div").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

d3.csv("res-mnist.csv", function (error, data) {
    if (error) throw error;

    // Convert string to int
    data.forEach(function (d) {
        d.id = Math.round(Number(+d.id));
        d.dim1 = +d.dim1;
        d.dim2 = +d.dim2;
        d.label = parseInt(d.label);
        d.p_x = +d.p_x;
        d.p_x_ = [+d.p_x_0, +d.p_x_1, +d.p_x_2, +d.p_x_3, +d.p_x_4,
        +d.p_x_5, +d.p_x_6, +d.p_x_7, +d.p_x_8, +d.p_x_9];
    });

    // Set x- and y-axis's range
    x.domain(d3.extent(data, function (d) { return d.dim1; })).nice();
    y.domain(d3.extent(data, function (d) { return d.dim2; })).nice();

    // Set x-axis's label
    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(xAxis)
        .append("text")
        .attr("class", "label")
        .attr("x", widthPlot)
        .attr("y", -6)
        .style("text-anchor", "end")
        .text("Dim 1");

    // Set y-axis's label
    svg.append("g")
        .attr("class", "y axis")
        .call(yAxis)
        .append("text")
        .attr("class", "label")
        .attr("transform", "rotate(-90)")
        .attr("y", 6)
        .attr("dy", ".71em")
        .style("text-anchor", "end")
        .text("Dim 2")

    // Make sactters
    svg.selectAll(".dot")
        .data(data)
        .enter().append("circle")
        .attr("class", "dot")
        .attr("r", 4)
        .attr("cx", function (d) { return x(d.dim1); })
        .attr("cy", function (d) { return y(d.dim2); })
        .style("fill", function (d) { return color(d.label); })
        .on("click", function (d) {
            legend.select("#p_x_c").remove();
            legend.append("text")
                .attr("id", "p_x_c")
                .attr("x", widthPlot + 64)
                .attr("y", 27)
                .attr("dy", ".35em")
                .style("text-anchor", "end")
                .text(function (i) { return d.p_x_[i].toFixed(5); });

            legend.selectAll("#p_x_c_rect").remove();
            legend.append("rect")
                .attr("id", "p_x_c_rect")
                .attr("y", 18)
                .attr("x", widthPlot + 81)
                .attr("width", function (i) { return d.p_x_[i] * 64; })
                .attr("height", 18)
                .style("fill", color);

            legend.append("rect")
                .attr("id", "p_x_c_rect")
                .attr("x", widthPlot + 81)
                .attr("y", 18)
                .attr("width", 64)
                .attr("height", 18)
                .style("opacity", 0.5)
                .style("fill", color)
                .style("stroke", "black")
                .style("stroke-width", function (i) { if (d.label == i) return 1; else return 0; });

            svg.select("#p_x").remove();
            svg.append("text")
                .attr("id", "p_x")
                .attr("x", 64 + 12)
                .attr("y", height - 9)
                .attr("dy", ".35em")
                .style("text-anchor", "end")
                .text(function () { return "" + d.p_x.toFixed(5); });

            svg.select("#x_img").remove();
            svg.append("image")
                .attr("id", "x_img")
                .attr("x", widthPlot + 32)
                .attr("y", 200 + 64)
                .attr('width', 112)
                .attr('height', 112)
                .attr("xlink:href", function () { return "images/mnist/test/" + d.id + ".jpg"; })
        });

    // Put legend
    var legend = svg.selectAll(".legend")
        .data(color.domain())
        .enter().append("g")
        .attr("class", "legend")
        .attr("transform", function (d, i) { return "translate(0," + (18 + i * 20) + ")"; });

    legend.append("rect")
        .attr("x", 32 + 8)
        .attr("y", 18)
        .attr("width", 18)
        .attr("height", 18)
        .style("fill", color);

    legend.append("text")
        .attr("x", 32)
        .attr("y", 18 + 9)
        .attr("dy", ".35em")
        .style("text-anchor", "end")
        .text(function (i) { return i; });

    // Bottom left default

    svg.append("text")
        .attr("x", 32)
        .attr("y", height - 9)
        .attr("dy", ".35em")
        .style("text-anchor", "end")
        .text("P(x) =");

    svg.append("text")
        .attr("id", "p_x")
        .attr("x", 64 - 4)
        .attr("y", height - 9)
        .attr("dy", ".35em")
        .style("text-anchor", "end")
        .text("?");

    // Right p_x_c default

    svg.append("text")
        .attr("x", widthPlot + 64 + 8)
        .attr("y", 18)
        .attr("dy", ".35em")
        .style("text-anchor", "end")
        .text("P(c = c' | x)");

    legend.append("text")
        .attr("id", "p_x_c")
        .attr("x", widthPlot + 64)
        .attr("y", 27)
        .attr("dy", ".35em")
        .style("text-anchor", "end")
        .text("?");

    legend.append("rect")
        .attr("id", "p_x_c_rect")
        .attr("x", widthPlot + 81)
        .attr("y", 18)
        .attr("width", 64)
        .attr("height", 18)
        .style("opacity", 0.5)
        .style("fill", color);

    legend.append("text")
        .attr("x", widthPlot + 128 + 32)
        .attr("y", 18 + 9)
        .attr("dy", ".35em")
        .style("text-anchor", "end")
        .text(function (i) { return i; });

    // Test

});