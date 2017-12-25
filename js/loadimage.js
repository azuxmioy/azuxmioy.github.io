$(document).ready(function () {
    $('#bigphoto').on('show.bs.modal', function (e) {
        var image = $(e.relatedTarget).attr('data-src');
        $("#large_disp").attr("src", image);
    });
});