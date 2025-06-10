window.dccFunctions = window.dccFunctions || {};
window.dccFunctions.toFixedTwo = function(value) {
    return Math.round(value * 100) / 100;
}