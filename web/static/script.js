$(document).on('ready', function() {
    $('form').on('submit', function(e) {
        var formData = new FormData($(this)[0]);
        $.ajax({
            url: $(this).attr('action'),
            type: 'POST',
            data: formData,
            success: function (data) {
                if (data.valid) {
                    $('#result').empty();
                    $.each(data.result, function(index, value) {
                        $('#result').append($('<img />', {src: value, class: "img-result"}));
                    });
                }
            },
            cache: false,
            contentType: false,
            processData: false
        });

        return false;
    });
});