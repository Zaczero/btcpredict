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
        <link href="/css/style.css" rel="stylesheet">
        
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

        <title>BTC Predict - Current bull run price and date prediction</title>
        <meta name="description" content="Bitcoin bull run prediction project which aims to evaluate current bull run's top price alongside the exact date." />
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-light">
            <a class="navbar-brand" href="/">BTCPredict</a>

            <ul class="navbar-nav ml-auto">
                <li class="nav-item active">
                    <a class="nav-link" href="https://github.com/Zaczero/btcpredict" target="_blank"><img class="mr-1" src="/img/github-icon.svg" height="16">View Source</a>
                </li>
            </ul>
        </nav>

        <div class="container-fluid">
        <div class="prediction">
            <p class="prediction-title">The top of the current BTC bull run will happen on:</p>
            <h1 class="prediction-price">$<?=$prediction_price?><wbr><span class="prediction-price-round">±$<?=$prediction_price_round?></span></h1>
            <h2 class="prediction-date"><nobr><?=$prediction_date?></nobr><wbr><span class="prediction-date-round">±<?=$prediction_date_round?> day(s)</span></h2>
            <p class="prediction-generated-on">This prediction was made on <nobr><?=$prediction_generated_one?></nobr></p>
            <p class="prediction-learn-more"><a href="https://github.com/Zaczero/btcpredict" target="_blank">Click here to learn more</a> 😎</p>
        </div>
        </div>
    </body>
</html>
