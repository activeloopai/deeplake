<template>
  <div :class="messageClasses" ref="message" v-if="visible">
    <div class="message__notice">
      <div class="message__content">
        <i :class="`message__icon message__icon--${type}`"></i>
        <span class="message__text">{{ message }}</span>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Message',
  data() {
    return {
      visible: true,
      animation: {
        enter: false,
        leave: false,
      },
    }
  },
  computed: {
    messageClasses() {
      const { enter, leave } = this.animation

      return {
        message: true,
        'move-up-enter': enter,
        'move-up-enter-active': enter,
        'move-up-leave': leave,
        'move-up-leave-active': leave,
      }
    },
  },
  methods: {
    enter() {
      this.animation.enter = true

      this.$refs.message.addEventListener('animationend', this.entered, false)
    },
    entered() {
      this.animation.enter = false

      this.$refs.message.removeEventListener(
        'animationend',
        this.entered,
        false
      )
    },
    leave() {
      this.animation.leave = true

      this.$refs.message.addEventListener('animationend', this.left, false)
    },
    left() {
      this.animation.leave = false

      this.$refs.message.removeEventListener('animationend', this.left, false)

      this.$refs.message.remove()
    },
  },
  mounted() {
    this.enter()

    setTimeout(() => {
      this.leave()
    }, 1000 * 3)
  },
}
</script>

<style lang="stylus">
#message__container
  position: fixed
  top: 16px
  left: 0
  width: 100%
  font-size: 14px
  line-height: 1.5
  pointer-events: none

.message
  padding: 8px
  margin-top: -8px
  text-align: center

  &__notice
    display: inline-block
    padding: 10px 16px
    color: #000000a6
    background: #fff
    border-radius: 4px
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15)
    pointer-events: all

  &__content
    display: flex
    align-items: center

  &__icon
    display: block
    width: 16px
    height: 16px
    margin-right: 8px
    background: no-repeat 0 0
    background-size: 100%

    &--info
      background-image: url("data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/PjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+PHN2ZyB0PSIxNTI2NTczODQxNDc4IiBjbGFzcz0iaWNvbiIgc3R5bGU9IiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjQ3NzgiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB3aWR0aD0iMjIwIiBoZWlnaHQ9IjIyMCI+PGRlZnM+PHN0eWxlIHR5cGU9InRleHQvY3NzIj48L3N0eWxlPjwvZGVmcz48cGF0aCBkPSJNMTAyNCA1MTJDMTAyNCAyMjkuMjM2IDc5NC43NjQgMCA1MTIgMFMwIDIyOS4yMzYgMCA1MTJzMjI5LjIzNiA1MTIgNTEyIDUxMiA1MTItMjI5LjIzNiA1MTItNTEyek01MjMuNjM2IDIzMi43MjdjMzIuMTE3IDAgNTguMTgyIDI2LjA2NiA1OC4xODIgNTguMTgycy0yNi4wNjUgNTguMTgyLTU4LjE4MiA1OC4xODJjLTMyLjExNiAwLTU4LjE4MS0yNi4wNjYtNTguMTgxLTU4LjE4MnMyNi4wNjUtNTguMTgyIDU4LjE4MS01OC4xODJ6TTM5NS42MzYgNzY4aDY5LjgxOVY0NDIuMTgyaC02OS44MTl2LTQ2LjU0NmgxODYuMTgyVjc2OGg2OS44MTh2NDYuNTQ1aC0yNTZWNzY4eiIgZmlsbD0iIzE4OGZmZiIgcC1pZD0iNDc3OSI+PC9wYXRoPjwvc3ZnPg==")

    &--success
      background-image: url("data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/PjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+PHN2ZyB0PSIxNTI2NTcyNzM5MjYxIiBjbGFzcz0iaWNvbiIgc3R5bGU9IiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjI1NDgiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB3aWR0aD0iMjIwIiBoZWlnaHQ9IjIyMCI+PGRlZnM+PHN0eWxlIHR5cGU9InRleHQvY3NzIj48L3N0eWxlPjwvZGVmcz48cGF0aCBkPSJNNTEyIDMyQzI0Ni40IDMyIDMyIDI0Ni40IDMyIDUxMnMyMTQuNCA0ODAgNDgwIDQ4MCA0ODAtMjE0LjQgNDgwLTQ4MFM3NzcuNiAzMiA1MTIgMzJ6IG0yMTQuNCA0MDBsLTIzMC40IDIzMC40Yy0xNiAxNi0zOC40IDE2LTU0LjQgMEwzMjAgNTQwLjhjLTE2LTE2LTE2LTM4LjQgMC01NC40czM4LjQtMTYgNTQuNCAwbDk2IDk2IDIwNC44LTIwNC44YzE2LTE2IDM4LjQtMTYgNTQuNCAwIDEyLjggMTYgMTIuOCAzOC40LTMuMiA1NC40eiIgZmlsbD0iIzUxYzQxYSIgcC1pZD0iMjU0OSI+PC9wYXRoPjwvc3ZnPg==")

    &--warning
      background-image: url("data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/PjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+PHN2ZyB0PSIxNTI2NTczOTk1NzA3IiBjbGFzcz0iaWNvbiIgc3R5bGU9IiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjU1NjUiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB3aWR0aD0iMjIwIiBoZWlnaHQ9IjIyMCI+PGRlZnM+PHN0eWxlIHR5cGU9InRleHQvY3NzIj48L3N0eWxlPjwvZGVmcz48cGF0aCBkPSJNNTEyIDY0QzI2NC42NCA2NCA2NCAyNjQuNjQgNjQgNTEyYzAgMjQ3LjQyNCAyMDAuNjQgNDQ4IDQ0OCA0NDggMjQ3LjQ4OCAwIDQ0OC0yMDAuNTc2IDQ0OC00NDhDOTYwIDI2NC42NCA3NTkuNDg4IDY0IDUxMiA2NHpNNTEyIDc2OGMtMjYuNDMyIDAtNDgtMjEuNTA0LTQ4LTQ4UzQ4NS41NjggNjcyIDUxMiA2NzJjMjYuNjI0IDAgNDggMjEuNTA0IDQ4IDQ4UzUzOC42MjQgNzY4IDUxMiA3Njh6TTU2MCA1MjhDNTYwIDU1NC41NiA1MzguNjI0IDU3NiA1MTIgNTc2IDQ4NS41NjggNTc2IDQ2NCA1NTQuNTYgNDY0IDUyOGwwLTIyNEM0NjQgMjc3LjQ0IDQ4NS41NjggMjU2IDUxMiAyNTZjMjYuNjI0IDAgNDggMjEuNDQgNDggNDhMNTYwIDUyOHoiIHAtaWQ9IjU1NjYiIGZpbGw9IiNmYWFkMTUiPjwvcGF0aD48L3N2Zz4=")

    &--error
      background-image: url("data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/PjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+PHN2ZyB0PSIxNTI2NTczNTcwNTI2IiBjbGFzcz0iaWNvbiIgc3R5bGU9IiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjMwNzAiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB3aWR0aD0iMjIwIiBoZWlnaHQ9IjIyMCI+PGRlZnM+PHN0eWxlIHR5cGU9InRleHQvY3NzIj48L3N0eWxlPjwvZGVmcz48cGF0aCBkPSJNNTEyIDI1LjZDMjQzLjM1MzYgMjUuNiAyNS42IDI0My4zNzkyIDI1LjYgNTEyczIxNy43NTM2IDQ4Ni40IDQ4Ni40IDQ4Ni40YzI2OC41OTUyIDAgNDg2LjQtMjE3Ljc3OTIgNDg2LjQtNDg2LjRTNzgwLjU5NTIgMjUuNiA1MTIgMjUuNnogbTIyOS45MTM2IDY1Mi4xODU2YTQ1LjM2MzIgNDUuMzYzMiAwIDAgMS02NC4xNTM2IDY0LjE1MzZMNTEyIDU3Ni4xNzkybC0xNjUuNzYgMTY1Ljc2YTQ1LjM4ODggNDUuMzg4OCAwIDAgMS02NC4xNzkyLTY0LjE1MzZMNDQ3Ljg0NjQgNTEybC0xNjUuNzg1Ni0xNjUuNzZhNDUuMzg4OCA0NS4zODg4IDAgMCAxIDY0LjE3OTItNjQuMTc5Mkw1MTIgNDQ3Ljg0NjRsMTY1Ljc4NTYtMTY1Ljc2YTQ1LjM2MzIgNDUuMzYzMiAwIDAgMSA2NC4xNTM2IDY0LjE1MzZMNTc2LjE3OTIgNTEybDE2NS43MzQ0IDE2NS43ODU2eiIgcC1pZD0iMzA3MSIgZmlsbD0iI2Y1MjMyZCI+PC9wYXRoPjwvc3ZnPg==")

.move-up-enter
  opacity: 0
  animation-timing-function: cubic-bezier(0.08, 0.82, 0.17, 1)
  animation-duration: 0.2s
  animation-fill-mode: both

.move-up-enter.move-up-enter-active
  animation-name: moveUpIn
  animation-play-state: running

.move-up-leave
  animation-timing-function: cubic-bezier(0.6, 0.04, 0.98, 0.34)
  animation-duration: 0.3s
  animation-fill-mode: both

.move-up-leave.move-up-leave-active
  animation-name: moveUpOut
  animation-play-state: running
  pointer-events: none

@keyframes moveUpIn {
  0% {
    transform-origin: 0 0
    transform: translateY(-100%)
    opacity: 0
  }

  to {
    transform-origin: 0 0
    transform: translateY(0)
    opacity: 1
  }
}

@keyframes moveUpOut {
  0% {
    transform-origin: 0 0
    transform: translateY(0)
    opacity: 1
  }

  to {
    height: 0
    transform-origin: 0 0
    transform: translateY(-100%)
    opacity: 0
  }
}
</style>
