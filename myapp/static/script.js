document.addEventListener('DOMContentLoaded', () => {
    const heading = document.querySelector('.blinking-heading');
    const text = heading.textContent;
    heading.innerHTML = text.split('').map(letter => `<span style="visibility: hidden;">${letter}</span>`).join('');

    const letters = heading.querySelectorAll('span');
    let index = 0;

    const intervalId = setInterval(() => {
        if (index < letters.length) {
            letters[index].style.visibility = 'visible';
            index++;
        } else {
            clearInterval(intervalId);
        }
    }, 60); // Show each letter every 1 second
});
