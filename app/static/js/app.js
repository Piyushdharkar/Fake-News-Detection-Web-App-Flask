
$(document).ready(function () {

    let chart = new Chart($("#probability_chart"), {
        type: 'bar',
        data: {
          labels: [],
          datasets: [
            {
              label: "Probability",
              backgroundColor: [],
              data: []
            }
          ]
        },
        options: {
          legend: { display: false },
          title: {
            display: true,
            text: 'Fake news probability'
          },
          scales: {
                yAxes : [{
                    ticks : {
                        max : 1,
                        min : 0
                    }
                }]
            }
        }
    });

    function generate_random_color () {
        return '#' + (Math.random().toString(16) + '0000000').slice(2, 8);
    };

    function update_chart(probabilities) {
        labels = [];
        ys = [];
        colors = [];

        $.each(probabilities, function(key, object) {
            $.each(object, function(key, value) {

                labels.push(value["label"]);
                colors.push(generate_random_color());
                ys.push(value["value"]);
            });
        });

        chart.data.labels = labels;
        chart.data.datasets[0].data = ys;
        chart.data.datasets[0].backgroundColor = colors;
        chart.update();
    }

    $("#post_news_article_text_button").click(function() {
        text = $("#news_article_text_area").val();
        title = $("#news_article_title").val();
        request_json = {'text':text, 'title':title};

        $.post('/predict', request_json).done(update_chart);
    });

});