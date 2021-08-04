<?php
define("DATA_FILE", "latest.json");
define("DATE_FORMAT", "F jS, Y");

if (!file_exists(DATA_FILE)) {
    http_response_code(500);
    exit("500: data file not found");
}

$json_txt = file_get_contents(DATA_FILE);
$json = json_decode($json_txt, true);

$prediction_price = number_format($json['predicted_price']);
$prediction_price_round = number_format($json['predicted_price_dev']);

$prediction_date_timestamp = $json['current_date'] + $json['predicted_date'] * 3600 * 24;
$prediction_date = date(DATE_FORMAT, $prediction_date_timestamp);
$prediction_date_round = $json['predicted_date_dev'];

$prediction_generated_one = date(DATE_FORMAT, $json['current_date']);
?><!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        
        <link rel="shortcut icon" href="/favicon.ico">
        <link rel="icon" sizes="16x16 32x32 64x64" href="/favicon.ico">
        <link rel="icon" type="image/png" sizes="196x196" href="/favicon-192.png">
        <link rel="icon" type="image/png" sizes="160x160" href="/favicon-160.png">
        <link rel="icon" type="image/png" sizes="96x96" href="/favicon-96.png">
        <link rel="icon" type="image/png" sizes="64x64" href="/favicon-64.png">
        <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32.png">
        <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16.png">
        <link rel="apple-touch-icon" href="/favicon-57.png">
        <link rel="apple-touch-icon" sizes="114x114" href="/favicon-114.png">
        <link rel="apple-touch-icon" sizes="72x72" href="/favicon-72.png">
        <link rel="apple-touch-icon" sizes="144x144" href="/favicon-144.png">
        <link rel="apple-touch-icon" sizes="60x60" href="/favicon-60.png">
        <link rel="apple-touch-icon" sizes="120x120" href="/favicon-120.png">
        <link rel="apple-touch-icon" sizes="76x76" href="/favicon-76.png">
        <link rel="apple-touch-icon" sizes="152x152" href="/favicon-152.png">
        <link rel="apple-touch-icon" sizes="180x180" href="/favicon-180.png">
        <meta name="msapplication-TileColor" content="#FFFFFF">
        <meta name="msapplication-TileImage" content="/favicon-144.png">
        <meta name="msapplication-config" content="/browserconfig.xml">

        <title>BTC Predict - Current Bitcoin bull run price and date prediction</title>
        <meta name="description" content="Bitcoin bull run prediction project which aims to evaluate current bull run's top price alongside the exact date." />

        <link rel="preload" href="/css/style.css" as="style" type="text/css">
        <link rel="preload" href="/font/Rubik-VariableFont_wght.ttf" as="font" type="font/ttf" crossorigin>
        <style>
        :root{--blue: #007bff;--indigo: #6610f2;--purple: #6f42c1;--pink: #e83e8c;--red: #dc3545;--orange: #fd7e14;--yellow: #ffc107;--green: #28a745;--teal: #20c997;--cyan: #17a2b8;--white: #fff;--gray: #6c757d;--gray-dark: #343a40;--primary: #007bff;--secondary: #6c757d;--success: #28a745;--info: #17a2b8;--warning: #ffc107;--danger: #dc3545;--light: #f8f9fa;--dark: #343a40;--breakpoint-xs: 0;--breakpoint-sm: 576px;--breakpoint-md: 768px;--breakpoint-lg: 992px;--breakpoint-xl: 1200px;--font-family-sans-serif: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";--font-family-monospace: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace}*,*::before,*::after{box-sizing:border-box}html{font-family:sans-serif;line-height:1.15;-webkit-text-size-adjust:100%}nav{display:block}body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,"Noto Sans","Liberation Sans",sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji";font-size:1rem;font-weight:400;line-height:1.5;color:#212529;text-align:left;background-color:#fff}h1,h2{margin-top:0;margin-bottom:.5rem}p{margin-top:0;margin-bottom:1rem}ul{margin-top:0;margin-bottom:1rem}a{color:#007bff;text-decoration:none;background-color:transparent}img{vertical-align:middle;border-style:none}::-webkit-file-upload-button{font:inherit;-webkit-appearance:button}h1,h2{margin-bottom:.5rem;font-weight:500;line-height:1.2}h1{font-size:2.5rem}h2{font-size:2rem}.container-fluid{width:100%;padding-right:15px;padding-left:15px;margin-right:auto;margin-left:auto}.nav-link{display:block;padding:.5rem 1rem}.navbar{position:relative;display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;padding:.5rem 1rem}.navbar-brand{display:inline-block;padding-top:.3125rem;padding-bottom:.3125rem;margin-right:1rem;font-size:1.25rem;line-height:inherit;white-space:nowrap}.navbar-nav{display:flex;flex-direction:column;padding-left:0;margin-bottom:0;list-style:none}.navbar-nav .nav-link{padding-right:0;padding-left:0}@media (min-width:992px){.navbar-expand-lg{flex-flow:row nowrap;justify-content:flex-start}.navbar-expand-lg .navbar-nav{flex-direction:row}.navbar-expand-lg .navbar-nav .nav-link{padding-right:.5rem;padding-left:.5rem}}.navbar-light .navbar-brand{color:rgba(0,0,0,.9)}.navbar-light .navbar-nav .nav-link{color:rgba(0,0,0,.5)}.navbar-light .navbar-nav .active>.nav-link{color:rgba(0,0,0,.9)}.prediction-price{margin-bottom:0!important}.mr-1{margin-right:.25rem!important}.prediction-generated-on{margin-bottom:.25rem!important}.prediction-date-round,.prediction-price-round{margin-left:1rem!important}.ml-auto{margin-left:auto!important}@font-face{font-family:"Rubik";font-display:swap;src:local("Rubik"),url("/font/Rubik-VariableFont_wght.ttf") format("truetype");unicode-range:U+000-5FF}html,body{font-family:"Rubik",sans-serif;min-height:100vh;background-color:#fff;background-attachment:fixed;background-size:cover}.navbar-brand{font-style:italic;font-weight:600}.prediction{margin:6rem 0;text-align:center}.prediction-title{font-size:34px;text-decoration:underline #31e981}.prediction-price{font-size:140px;font-weight:600;color:#31e981;text-shadow:1px 2px 0 #28666e;line-height:1}.prediction-price-round{font-size:95px;color:#7d83ff;white-space:nowrap}.prediction-date{margin-bottom:5rem;font-size:50px}.prediction-date-round{font-size:34px;white-space:nowrap}.prediction-generated-on{font-size:20px;color:#888;letter-spacing:.5px}.prediction-learn-more{font-size:16px}@media (max-width:1240px){.prediction{margin:5rem 0}.prediction-title{font-size:24px}.prediction-price{margin-bottom:1rem!important;font-size:94px}.prediction-price-round{font-size:64px}.prediction-date{margin-bottom:4rem;font-size:32px}.prediction-date-round{font-size:24px}}@media (max-width:840px){.prediction{margin:4rem 0}.prediction-title{font-size:20px}.prediction-price{font-size:80px}.prediction-price-round{font-size:52px}.prediction-date{margin-bottom:3rem;font-size:26px}.prediction-date-round{font-size:22px}}@media (max-width:520px){.prediction{margin:3rem 0}.prediction-title{font-size:16px}.prediction-price{font-size:68px}.prediction-price-round{margin-left:.5rem!important;font-size:48px}.prediction-date{margin-bottom:2rem;font-size:22px}.prediction-date-round{margin-left:.5rem!important;font-size:18px}.prediction-generated-on{font-size:16px}.prediction-learn-more{font-size:14px}}
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-light">
            <a class="navbar-brand" href="/">BTCPredict</a>
            <ul class="navbar-nav ml-auto">
                <li class="nav-item active">
                    <a class="nav-link" href="https://github.com/Zaczero/btcpredict" target="_blank"><img class="mr-1" src="/img/github-icon.svg" width="16" height="16">View Source</a>
                </li>
            </ul>
        </nav>

        <div class="container-fluid">
        <div class="prediction">
            <p class="prediction-title">The top of the current BTC bull run is:</p>
            <h1 class="prediction-price">$<?=$prediction_price?><wbr><span class="prediction-price-round">±$<?=$prediction_price_round?></span></h1>
            <h2 class="prediction-date"><nobr><?=$prediction_date?></nobr><wbr><span class="prediction-date-round">±<?=$prediction_date_round?> day(s)</span></h2>
            <p class="prediction-generated-on">This prediction was made on <nobr><?=$prediction_generated_one?></nobr></p>
            <p class="prediction-learn-more"><a href="https://github.com/Zaczero/btcpredict" target="_blank">Click here to learn more</a> 😎</p>
        </div>

        <div class="announcement mb-5">
            <div class="alert alert-primary col-lg-8 offset-lg-2 col-xl-6 offset-xl-3">
                <h5 class="alert-heading"><b>Notice:</b> Some people are getting worried that the predicted date is too early.</h5>

                <p>
                    The current date model relies heavily on The Golden 51%-49% Ratio metric which may or may not be correct.
                    If it happens so that the bull run peak does not occur on the predicted date, a new model will be engineered with exclusion to faulty metrics.
                </p>

                <hr>

                <p class="mb-0">
                    For extra bull run peak reassurance I recommend checking the CBBI score (my other project): <a href="https://cbbi.info" target="_blank" rel="noreferrer">https://cbbi.info</a>
                </p>
            </div>
        </div>
        </div>

        <link rel="stylesheet" href="/css/style.css" type="text/css">
    </body>
</html>
