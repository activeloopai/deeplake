import Vue from 'vue'
import Main from './message.vue'

const MessageConstructor = Vue.extend(Main)

const typeList = ['info', 'success', 'warning', 'error']

const getMessageContainerElement = () => {
  let el = document.body.querySelector('#message__container')

  if (!el) {
    el = document.createElement('div')
    el.id = 'message__container'
    document.body.appendChild(el)
  }

  return el
}

const Message = options => {
  if (options === undefined) return

  const opts =
    typeof options === 'string' ? { type: 'info', message: options } : options

  const vm = new MessageConstructor({
    data: opts,
  }).$mount()

  const messageContainerElement = getMessageContainerElement()

  messageContainerElement.appendChild(vm.$el)

  vm.visible = true

  return vm
}

typeList.forEach(type => {
  Message[type] = options => {
    const opts = typeof options === 'string' ? { message: options } : options

    opts.type = type

    return Message(opts)
  }
})

export default Message
